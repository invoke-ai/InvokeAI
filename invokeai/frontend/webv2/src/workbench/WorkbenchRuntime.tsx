import type { ReactNode } from 'react';

import { galleryBoardExists } from '@features/gallery/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { useState } from 'react';

import type { WorkbenchInternalStore } from './workbenchStore';

import { createCanvasDimsSync } from './canvasDimsSync';
import { createWorkbenchFocusController, FocusRegionProvider } from './focusRegions';
import { createShortcutHintSources, ShortcutHintSourcesProvider } from './hotkeys/hintSources';
import { useWorkbenchInternalStore, useWorkbenchQueries, useWorkbenchSubscription } from './WorkbenchContext';
import { getProjectGalleryBoardReferences } from './workbenchState';

/** Workbench-owned lifecycle adapter for aggregate-local synchronization. Cross-module adapters are constructed by App. */
export const WorkbenchRuntime = () => {
  const store = useWorkbenchInternalStore();

  useMountEffect(() => {
    const canvasDimsSync = createCanvasDimsSync(store);
    const boardReferenceCheck = createGalleryBoardReferenceCheck(store);

    return () => {
      canvasDimsSync.dispose();
      boardReferenceCheck.dispose();
    };
  });

  return null;
};

/**
 * Clears what each project loaded into the workbench says about boards that are gone. A board can go while the project
 * is not open — deleted from another tab, or with a project deleted from the Launchpad — and the project's saved
 * selection, auto-add board and unfinished queue items would keep naming it, sending results to a board that no longer
 * exists. Each project is checked as it is loaded, against what the server says then: an earlier answer for the same
 * board may be stale, since it can go with a project or board deleted in between. Only a question still in flight is
 * shared, and a project closed and opened again is checked again. The Launchpad is a route of its own, so returning
 * from it mounts a new workbench, which checks every project it loads.
 */
export const createGalleryBoardReferenceCheck = (
  store: Pick<WorkbenchInternalStore, 'commands' | 'getSnapshot' | 'subscribe'>,
  boardExists: (boardId: string, signal?: AbortSignal) => Promise<boolean> = galleryBoardExists
): { dispose: () => void } => {
  const owner = captureAccountScope();
  const lifetime = new AbortController();
  const signal = AbortSignal.any([owner.signal, lifetime.signal]);
  const checkedProjectIds = new Set<string>();
  const unansweredBoardIds = new Set<string>();
  let checkedProjects: unknown = null;

  const ask = (boardId: string) => {
    unansweredBoardIds.add(boardId);
    boardExists(boardId, signal)
      .then((exists) => {
        if (!exists && !signal.aborted && isAccountScopeCurrent(owner)) {
          store.commands.gallery.reconcileDeletedBoardOutcome({
            boardId,
            deletedBoardImageNames: [],
            deletedBoardVideoNames: [],
            deletedImageNames: [],
            deletedVideoNames: [],
            failedImageNames: [],
            failedVideoNames: [],
          });
        }
      })
      // Unanswered says nothing about the board; it stays named.
      .catch(() => undefined)
      .finally(() => unansweredBoardIds.delete(boardId));
  };

  // Every edit notifies; only a change to the set of loaded projects can bring one in to check.
  const check = () => {
    const { hasHydrated, projects } = store.getSnapshot();

    if (!hasHydrated || projects === checkedProjects || signal.aborted) {
      return;
    }

    checkedProjects = projects;
    const loadedProjectIds = new Set(projects.map((project) => project.id));

    for (const projectId of checkedProjectIds) {
      if (!loadedProjectIds.has(projectId)) {
        checkedProjectIds.delete(projectId);
      }
    }

    for (const project of projects) {
      if (checkedProjectIds.has(project.id)) {
        continue;
      }

      checkedProjectIds.add(project.id);
      getProjectGalleryBoardReferences(project)
        .filter((boardId) => !unansweredBoardIds.has(boardId))
        .forEach(ask);
    }
  };

  check();
  const unsubscribe = store.subscribe(check);

  return {
    dispose: () => {
      unsubscribe();
      lifetime.abort();
    },
  };
};

/**
 * Owns workbench focus for everything below it — the shell and the hotkey runtime read the same target. Focus is
 * transient: it is forgotten, along with any focus move still in flight, when the project on screen changes, the
 * account changes, or the workbench unmounts, and a window's focus is forgotten once it docks or closes.
 *
 * It lives in this module, beside the other workbench lifecycle adapter, because the editor's startup module set
 * is pinned by the architecture performance gate; a module of its own would have to be added to that baseline.
 */
export const WorkbenchFocusProvider = ({ children }: { children: ReactNode }) => {
  const { getSnapshot } = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();
  const [hintSources] = useState(createShortcutHintSources);
  const [controller] = useState(() =>
    createWorkbenchFocusController({
      getProjectId: () => getSnapshot().activeProject.id,
      isFloating: (instanceId) => getSnapshot().activeProject.floatingWidgets?.[instanceId] !== undefined,
    })
  );

  useMountEffect(() => {
    const clear = () => {
      controller.clear();
      hintSources.focus.clear();
    };
    let projectId = getSnapshot().activeProject.id;
    const unsubscribe = subscribe(() => {
      const nextProjectId = getSnapshot().activeProject.id;

      if (nextProjectId !== projectId) {
        projectId = nextProjectId;
        clear();
      } else {
        controller.forgetClosedWindow();
      }
    });
    const unregister = registerAccountOwnedResource({ clear, name: 'workbench-focus' });

    return () => {
      unsubscribe();
      unregister();
      clear();
    };
  });

  return (
    <FocusRegionProvider controller={controller}>
      <ShortcutHintSourcesProvider sources={hintSources}>{children}</ShortcutHintSourcesProvider>
    </FocusRegionProvider>
  );
};
