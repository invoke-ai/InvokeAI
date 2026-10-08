import type { ProjectLayoutState } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type { ProjectSettings } from '@workbench/settings/contracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { startIntermediatesHoldLease } from '@features/intermediates/holdLease';
import { createLogger } from '@platform/logging/logger';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { useDebouncedValue } from '@platform/react/useDebouncedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
import { shallowEqual as selectorShallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { createContext, use, useSyncExternalStore, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectPushOutcome } from './projects/projectFlush';

import { WorkbenchSplashScreen } from './components/WorkbenchSplashScreen';
import { WorkbenchUnavailableScreen } from './components/WorkbenchUnavailableScreen';
import { createExtensionRegistry, type ExtensionRegistry } from './extensions/extensionRegistry';
import { clearLayerPanelStates } from './layerPanelState';
import { browserPageLifecycle, createWorkbenchPersistenceRuntime, type PersistenceExit } from './persistenceRuntime';
import { createOpenProjectBroker } from './projects/openProjectBroker';
import {
  createLiveCanvasEngines,
  createOpenProjectsHeldMediaReader,
  type LiveCanvasEngines,
} from './projects/projectAssets';
import { describeRefusedProjects } from './projects/projectLoadRefusal';
import {
  createSyncedWorkbenchPersistence,
  type SyncedWorkbenchPersistence,
  type WorkbenchLoadOptions,
} from './projects/syncedPersistence';
import { consumeWorkspaceClearFailure } from './settings/clearWorkspaceData';
import { getProjectWidgetValues } from './widgetState';
import { createWorkbenchStore, type WorkbenchSnapshot, type WorkbenchInternalStore } from './workbenchStore';

interface WorkbenchContextValue {
  activeProject: Project;
  /** Queue side effects must wait for hydration to avoid acting on state about to be replaced. */
  hasHydrated: boolean;
}

type EqualityFn<T> = (left: T, right: T) => boolean;
type WorkbenchSelector<T> = (snapshot: WorkbenchSnapshot) => T;

const WorkbenchStoreContext = createContext<WorkbenchInternalStore | null>(null);
const WorkbenchPersistenceContext = createContext<SyncedWorkbenchPersistence | null>(null);
const WorkbenchExtensionsContext = createContext<ExtensionRegistry | null>(null);
const WorkbenchLiveCanvasEnginesContext = createContext<LiveCanvasEngines | null>(null);
const subscribeToNothing = (): (() => void) => () => {};
const getNullSnapshot = (): null => null;

/**
 * Each account lifetime's latest editor exit in this tab. The next editor loads only after it settles (bounded by the
 * runtime), so it never hydrates a snapshot older than the checkpoint and two editors never write at once. Keyed by
 * the lifetime: another account neither waits for it nor sees it.
 */
const pendingEditorExits = new WeakMap<AccountScope, PersistenceExit>();

export const shallowEqual = selectorShallowEqual;

export const WorkbenchProvider = ({
  children,
  loadOptions,
}: {
  children: ReactNode;
  /** Boot-time session options (deep-linked project, fresh draft). Read once at mount. */
  loadOptions?: WorkbenchLoadOptions;
}) => {
  const [store] = useState(() => createWorkbenchStore());
  const [owner] = useState(captureAccountScope);
  const { t } = useTranslation();
  const [persistence] = useState(() => createSyncedWorkbenchPersistence(owner));
  const [extensions] = useState(createExtensionRegistry);
  const [liveCanvasEngines] = useState(createLiveCanvasEngines);
  const [loadUnavailable, setLoadUnavailable] = useState<{ message: string; retry(): void } | null>(null);
  const hasHydrated = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot).hasHydrated;

  // The runtime is created inside the effect: exit is terminal, so each
  // mount (including a StrictMode remount) must get its own instance.
  useMountEffect(() => {
    const releasePersistence = persistence.retain();
    const persistenceRuntime = createWorkbenchPersistenceRuntime({
      aggregate: {
        ...store.internal.persistence,
        getPersistedRevision: store.getPersistedRevision,
        notifyProjectNotFound: () =>
          store.commands.notifications.add({
            kind: 'info',
            message: 'The linked project does not exist on this account — it may have been deleted.',
            title: 'Project not found',
          }),
        reportLoadAvailable: () => {
          setLoadUnavailable(null);
          const clearFailure = consumeWorkspaceClearFailure(window.sessionStorage);
          if (clearFailure) {
            store.commands.notifications.reportError({
              area: 'workspace-clear',
              message: clearFailure,
              namespace: 'system',
            });
          }
        },
        reportLoadError: (message) =>
          store.commands.notifications.reportError({ area: 'persistence-load', message, namespace: 'system' }),
        reportLoadUnavailable: (message) =>
          setLoadUnavailable({ message, retry: () => persistenceRuntime.retryLoad() }),
        reportRefusedProjects: (refused) => {
          const notice = describeRefusedProjects(refused, t);

          if (notice) {
            store.commands.notifications.add({ kind: 'info', ...notice });
          }
        },
        setHasHydrated: store.setHasHydrated,
        subscribe: store.subscribe,
      },
      // At exit, React runs a removed provider's cleanup before its descendants', so their flushers are still registered.
      commitDrafts: flushWorkbenchDrafts,
      loadOptions,
      logger: createLogger({ area: 'autosave', namespace: 'persistence' }, { owner }),
      page: browserPageLifecycle,
      persistence,
      previousExit: pendingEditorExits.get(owner),
      signal: owner.signal,
    });
    // Publish throughout the mount so sibling library surfaces mutate open projects through the sync engine.
    const openProjectBroker = createOpenProjectBroker({
      closeProject: (projectId) => {
        // Skip close's last-tab refusal during deletion; leaveEditorIfLast owns leaving the editor.
        if (store.getSnapshot().projects.length > 1) {
          store.commands.projects.close(projectId);
        }
      },
      deleteProject: (projectId) => persistence.deleteProjectOnServer(projectId),
      flushPixels: (projectId) => liveCanvasEngines.flushPendingPixels(projectId),
      flushProject: (projectId) => {
        const project = store.getSnapshot().projects.find((candidate) => candidate.id === projectId);

        // Unopened projects have no local edits; their ids reflect server acknowledgements.
        return project
          ? persistence.flushProjectToServer(project)
          : Promise.resolve<ProjectPushOutcome>({ documentJson: '', kind: 'acknowledged' });
      },
      getOpenProjectIds: () => store.getSnapshot().projects.map((project) => project.id),
      getProject: (projectId) => store.getSnapshot().projects.find((candidate) => candidate.id === projectId),
      markProjectDeleted: (projectId) => {
        persistence.markProjectDeleted(projectId);
      },
      renameProject: (projectId, name) => {
        store.commands.projects.rename(projectId, name);
      },
      subscribe: store.subscribe,
      unmarkProjectDeleted: (projectId) => {
        persistence.unmarkProjectDeleted(projectId);
      },
    });

    persistenceRuntime.start();
    const releaseIntermediateHold = startIntermediatesHoldLease({
      owner,
      read: createOpenProjectsHeldMediaReader(() => store.getSnapshot().projects, liveCanvasEngines.heldAssets),
      subscribe: (onChange) => {
        let projects = store.getSnapshot().projects;
        const unsubscribeProjects = store.subscribe(() => {
          const next = store.getSnapshot().projects;
          if (next !== projects) {
            projects = next;
            onChange();
          }
        });
        const unsubscribeCanvas = liveCanvasEngines.subscribe(onChange);
        return () => {
          unsubscribeProjects();
          unsubscribeCanvas();
        };
      },
    });

    return () => {
      releaseIntermediateHold();
      clearLayerPanelStates();
      openProjectBroker.dispose();
      // The checkpoint outlives this unmount and keeps the persistence lease until it settles.
      const exit = persistenceRuntime.exit({
        beforeCapture: async () => {
          const flushes = await Promise.allSettled(
            store.getSnapshot().projects.map((project) => liveCanvasEngines.flushPendingPixels(project.id))
          );
          const failures = flushes.flatMap((flush) => (flush.status === 'rejected' ? [flush.reason] : []));
          if (failures.length > 0) {
            throw new AggregateError(failures, 'Unsaved Canvas pixels could not be persisted.');
          }
        },
      });
      const pendingExit: PersistenceExit = {
        settled: exit.settled.then(releasePersistence),
        supersede: () => {
          exit.supersede();
          // The lease itself stays until a stalled request returns: closing now could cut off a write in progress.
          persistence.releaseMutationLocks();
        },
      };
      pendingEditorExits.set(owner, pendingExit);
      void pendingExit.settled.then(() => {
        if (pendingEditorExits.get(owner) === pendingExit) {
          pendingEditorExits.delete(owner);
        }
      });
    };
  });

  return (
    <WorkbenchPersistenceContext value={persistence}>
      <WorkbenchLiveCanvasEnginesContext value={liveCanvasEngines}>
        <WorkbenchExtensionsContext value={extensions}>
          <WorkbenchStoreContext value={store}>
            {loadUnavailable ? (
              <WorkbenchUnavailableScreen
                message={loadUnavailable.message}
                onRetry={loadUnavailable.retry}
                persistence={persistence}
              />
            ) : hasHydrated ? (
              children
            ) : (
              <WorkbenchSplashScreen messageKey="splash.openingProject" />
            )}
          </WorkbenchStoreContext>
        </WorkbenchExtensionsContext>
      </WorkbenchLiveCanvasEnginesContext>
    </WorkbenchPersistenceContext>
  );
};

const useWorkbenchStore = (): WorkbenchInternalStore => {
  const store = use(WorkbenchStoreContext);

  if (!store) {
    throw new Error('useWorkbenchStore must be used within a WorkbenchProvider.');
  }

  return store;
};

const useOptionalWorkbenchStore = (): WorkbenchInternalStore | null => use(WorkbenchStoreContext);

/** Privileged aggregate adapter for persistence and resource-owning runtimes only. */
export const useWorkbenchInternalStore = (): WorkbenchInternalStore => useWorkbenchStore();

export const useHasWorkbenchProvider = (): boolean => useOptionalWorkbenchStore() !== null;

export const useWorkbenchSelector = <Selected,>(
  selector: WorkbenchSelector<Selected>,
  isEqual: EqualityFn<Selected> = shallowEqual
): Selected => {
  const store = useWorkbenchStore();

  return useExternalStoreSelector(store.subscribe, store.getSnapshot, selector, isEqual);
};

/** Debounce a selection; see `useDebouncedValue` for `settlesImmediately`. */
export const useDebouncedWorkbenchSelector = <Selected,>(
  selector: WorkbenchSelector<Selected>,
  debounceMs = 300,
  isEqual: EqualityFn<Selected> = Object.is,
  settlesImmediately?: (previous: Selected, next: Selected) => boolean
): Selected => useDebouncedValue(useWorkbenchSelector(selector, isEqual), debounceMs, { isEqual, settlesImmediately });

export const useActiveProject = (): Project => useWorkbenchSelector((snapshot) => snapshot.activeProject);

export const useActiveProjectSelector = <Selected,>(
  selector: (project: Project) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useWorkbenchSelector((snapshot) => selector(snapshot.activeProject), isEqual);

export const useActiveProjectId = (): string => useWorkbenchSelector((snapshot) => snapshot.activeProject.id);

export const useActiveProjectName = (): string => useActiveProjectSelector((project) => project.name);

export const useActiveProjectLayoutSelector = <Selected,>(
  selector: (layout: ProjectLayoutState) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useActiveProjectSelector((project) => selector(project.layout), isEqual);

export const useActiveProjectSettingsSelector = <Selected,>(
  selector: (settings: ProjectSettings) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useActiveProjectSelector((project) => selector(project.settings), isEqual);

export const useWidgetValuesSelector = <Selected,>(
  widgetId: WidgetTypeId,
  selector: (values: Record<string, unknown>) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useActiveProjectSelector((project) => selector(getProjectWidgetValues(project, widgetId)), isEqual);

export const useWidgetInstanceValuesSelector = <Selected,>(
  instanceId: WidgetInstanceId,
  selector: (values: Record<string, unknown>) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected =>
  useActiveProjectSelector((project) => selector(project.widgetInstances[instanceId]?.state.values ?? {}), isEqual);

export const useProjectWidgetInstanceValuesSelector = <Selected,>(
  projectId: string,
  instanceId: WidgetInstanceId,
  selector: (values: Record<string, unknown>) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected =>
  useWorkbenchSelector((snapshot) => {
    const project = snapshot.projects.find((candidate) => candidate.id === projectId);

    return selector(project?.widgetInstances[instanceId]?.state.values ?? {});
  }, isEqual);

export const useWorkbenchHasHydrated = (): boolean => useWorkbenchSelector((snapshot) => snapshot.hasHydrated);

/** Stable intent-oriented aggregate commands; callers never receive reducer actions. */
export const useWorkbenchCommands = () => useWorkbenchStore().commands;

export const useWorkbenchQueries = () => useWorkbenchStore().queries;

/** Stable read-model subscription used by external runtime adapters. */
export const useWorkbenchSubscription = () => useWorkbenchStore().subscribe;

/** Privileged persistence read/write port; this is the only UI-adjacent full-state adapter. */
export const useWorkbenchPersistenceAdapter = () => useWorkbenchStore().internal.persistence;

export const useWorkbenchPersistenceService = (): SyncedWorkbenchPersistence => {
  const persistence = use(WorkbenchPersistenceContext);
  if (!persistence) {
    throw new Error('useWorkbenchPersistenceService must be used within a WorkbenchProvider.');
  }
  return persistence;
};

export const useOptionalWorkbenchPersistenceService = (): SyncedWorkbenchPersistence | null =>
  use(WorkbenchPersistenceContext);

/** Where live Canvas engines register the undo state the hold lease must keep. */
export const useWorkbenchLiveCanvasEngines = (): LiveCanvasEngines => {
  const liveEngines = use(WorkbenchLiveCanvasEnginesContext);
  if (!liveEngines) {
    throw new Error('useWorkbenchLiveCanvasEngines must be used within a WorkbenchProvider.');
  }
  return liveEngines;
};

export const useWorkbenchExtensions = (): ExtensionRegistry => {
  const extensions = use(WorkbenchExtensionsContext);
  if (!extensions) {
    throw new Error('useWorkbenchExtensions must be used within a WorkbenchProvider.');
  }
  return extensions;
};

export const useOptionalWorkbenchExtensions = (): ExtensionRegistry | null => use(WorkbenchExtensionsContext);

export const useOptionalWorkbenchCommands = () => useOptionalWorkbenchStore()?.commands ?? null;

export const useOptionalWorkbenchQueries = () => useOptionalWorkbenchStore()?.queries ?? null;

export const useOptionalWorkbenchSelector = <Selected,>(
  selector: WorkbenchSelector<Selected>,
  fallback: Selected,
  isEqual: EqualityFn<Selected> = shallowEqual
): Selected => {
  const store = useOptionalWorkbenchStore();

  return useExternalStoreSelector(
    store?.subscribe ?? subscribeToNothing,
    store?.getSnapshot ?? getNullSnapshot,
    (snapshot) => (snapshot ? selector(snapshot) : fallback),
    isEqual
  );
};

export const useWorkbench = (): WorkbenchContextValue => {
  const store = useWorkbenchStore();
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  return snapshot;
};

export const useOptionalWorkbench = (): WorkbenchContextValue | null => {
  const store = useOptionalWorkbenchStore();
  const snapshot = useSyncExternalStore(
    store?.subscribe ?? subscribeToNothing,
    store?.getSnapshot ?? getNullSnapshot,
    store?.getSnapshot ?? getNullSnapshot
  );

  return store && snapshot ? snapshot : null;
};
