import type { AccountScope } from '@platform/state/accountLifecycle';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { LiveCanvasEngines } from './projectAssets';
import type { SyncedWorkbenchPersistence } from './syncedPersistence';

export class WorkbenchReloadError extends Error {
  constructor(
    readonly reason: 'inactive' | 'unsaved' | 'changed' | 'unavailable',
    options?: ErrorOptions
  ) {
    super(`Workbench reload refused: ${reason}`, options);
    this.name = 'WorkbenchReloadError';
  }
}

/** User-facing recovery copy is localized; diagnostic causes stay on the thrown error. */
export const describeWorkbenchReloadError = (error: unknown, t: (key: string) => string): string =>
  t(
    `widgets.failure.reloadReasons.${error instanceof WorkbenchReloadError ? error.reason : error instanceof AggregateError ? 'pixels' : 'failed'}`
  );

/** A fresh document re-reads the deployment's module graph after a repeated import rejection. Never leave edits behind to obtain it. */
export const reloadWorkbench = async ({
  owner,
  isCurrent,
  store,
  persistence,
  engines,
  reload,
}: {
  owner: AccountScope;
  isCurrent: () => boolean;
  store: Pick<WorkbenchInternalStore, 'getState' | 'getSnapshot' | 'getPersistedRevision' | 'internal'>;
  persistence: Pick<SyncedWorkbenchPersistence, 'saveWorkbench' | 'hasClosedSession'>;
  engines: Pick<LiveCanvasEngines, 'flushPendingPixels' | 'subscribe'>;
  reload: () => void;
}) => {
  const assertCurrent = () => {
    if (!isAccountScopeCurrent(owner) || !isCurrent() || persistence.hasClosedSession()) {
      throw new WorkbenchReloadError('inactive');
    }
  };
  assertCurrent();
  flushWorkbenchDrafts();
  const pixels = await Promise.allSettled(
    store.getSnapshot().projects.map((project) => engines.flushPendingPixels(project.id))
  );
  const failures = pixels.flatMap((result) => (result.status === 'rejected' ? [result.reason] : []));
  if (failures.length) {
    throw new AggregateError(failures, 'Unsaved Canvas pixels could not be persisted.');
  }
  assertCurrent();
  flushWorkbenchDrafts();
  const revision = store.getPersistedRevision();
  let canvasChanged = false;
  const unsubscribe = engines.subscribe(() => {
    canvasChanged = true;
  });
  try {
    const result = await persistence.saveWorkbench(store.getState());
    // Server board identities survive a refused reload, as they do a stale autosave result.
    for (const assignment of result.projectBoardAssignments) {
      if (!isAccountScopeCurrent(owner)) {
        break;
      }
      store.internal.persistence.assignProjectBoard(assignment);
    }
    assertCurrent();
    if (result.error || result.hasPendingChanges || result.awaitsDecision) {
      throw new WorkbenchReloadError('unsaved', { cause: result.error });
    }
    // Like project close, cross the paint barrier again after the push. The visible project is last so a held
    // gesture cannot start there while another project's barrier is pending, then disappear into a reload.
    const { activeProject, projects } = store.getSnapshot();
    const ids = projects
      .map((project) => project.id)
      .sort((a, b) => Number(a === activeProject.id) - Number(b === activeProject.id));
    for (const id of ids) {
      assertCurrent();
      try {
        await engines.flushPendingPixels(id);
      } catch (error) {
        throw new AggregateError([error], 'Unsaved Canvas pixels could not be persisted.');
      }
    }
    assertCurrent();
    flushWorkbenchDrafts();
    if (canvasChanged || store.getPersistedRevision() !== revision) {
      throw new WorkbenchReloadError('changed');
    }
    reload();
  } finally {
    unsubscribe();
  }
};
