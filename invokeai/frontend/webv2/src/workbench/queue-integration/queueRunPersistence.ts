import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import { type AccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { QueueRecallCache, QueueRecallSnapshotInput } from './queueRecallCache';
import type { QueueRunJournal } from './queueRunJournal';

export const withQueueRecallCache = <T extends Pick<QueueRunJournal, 'record' | 'settle'>>(
  journal: T,
  cache: Pick<QueueRecallCache, 'put'>,
  getItem: (projectId: string, queueItemId: string) => WorkbenchQueueItem | undefined
): T => {
  const pendingRecall = new Map<string, QueueRecallSnapshotInput>();
  const keyOf = (projectId: string, queueItemId: string) => JSON.stringify([projectId, queueItemId]);
  const remember = (projectId: string, item: WorkbenchQueueItem): void => {
    const { recall, sourceId, submittedAt } = item.snapshot;
    if (recall?.generateValues || recall?.videoValues) {
      pendingRecall.set(keyOf(projectId, item.id), {
        ...recall,
        projectId,
        queueItemId: item.id,
        sourceId,
        submittedAt,
      });
    }
  };
  return {
    ...journal,
    record(projectId, item) {
      const liveItem = item as WorkbenchQueueItem;
      remember(projectId, liveItem);
      return journal.record(projectId, item);
    },
    async settle(projectId, queueItemId) {
      const key = keyOf(projectId, queueItemId);
      const item = getItem(projectId, queueItemId);
      if (item) {
        remember(projectId, item);
      }
      const snapshot = pendingRecall.get(key);
      if (snapshot) {
        await cache.put(snapshot).catch(() => undefined);
        pendingRecall.delete(key);
      }
      return journal.settle(projectId, queueItemId);
    },
  };
};

type RunProject = Pick<Project, 'queue' | 'workflows'>;

/**
 * The library template whose `last_run_at` a routed result should bump: the run must have just turned completed
 * (a cancelled run, or a second routing of the same completion, bumps nothing) and come from a project workflow
 * linked to a library template. Dependency-free so the editor boot path never loads the workflow data module; the
 * caller drops bundled templates once that module loads. On a multiuser server (`viewerUserId` set) a template whose recorded
 * owner is another account is skipped; a link without a recorded owner is sent, and the server refuses it (403)
 * unless the caller owns the template or is an administrator, whose run then bumps the owner's row.
 */
export const getCompletedLibraryWorkflowRun = (
  before: RunProject | undefined,
  after: RunProject | undefined,
  queueItemId: string,
  viewerUserId: string | null
): string | null => {
  const previous = before?.queue.items.find((item) => item.id === queueItemId);
  const queueItem = after?.queue.items.find((item) => item.id === queueItemId);
  const submission = queueItem?.snapshot.backendSubmission;

  if (
    !after ||
    !queueItem ||
    queueItem.status !== 'completed' ||
    previous?.status === 'completed' ||
    submission?.kind !== 'workflow' ||
    !submission.projectWorkflowId
  ) {
    return null;
  }

  const source = after.workflows.entries.find((entry) => entry.document.id === submission.projectWorkflowId)?.source;

  if (!source || (viewerUserId !== null && source.ownerUserId !== undefined && source.ownerUserId !== viewerUserId)) {
    return null;
  }

  return source.libraryWorkflowId;
};

export interface LibraryWorkflowRunTouchOptions {
  getProject(projectId: string): RunProject | undefined;
  isDisposed(): boolean;
  /** The account the queue runtime runs for; nothing is sent once it is no longer current. */
  owner: AccountScope;
  /** The signed-in user on a multiuser server; null in single-user mode. */
  viewerUserId: string | null;
}

/**
 * Routes a run's results, then marks the template it ran from as run (fire and forget). Ordering by last run is a
 * convenience: nothing here can throw into the queue path, and a failed or stale request is dropped. The workflow
 * data module loads only for a completed linked run, so it stays out of the editor's startup bundle.
 */
export const withLibraryWorkflowRunTouch =
  <Payload extends { projectId: string; queueItemId: string }>(
    routeResults: (payload: Payload) => void,
    { getProject, isDisposed, owner, viewerUserId }: LibraryWorkflowRunTouchOptions
  ) =>
  (payload: Payload): void => {
    let before: RunProject | undefined | null = null;

    try {
      before = getProject(payload.projectId);
    } catch {
      // Without the state before routing a repeat cannot be told from a completion; skip the bump.
    }

    routeResults(payload);

    if (before === null) {
      return;
    }

    const isCurrent = () => !isDisposed() && isAccountScopeCurrent(owner);
    let libraryWorkflowId: string | null = null;

    try {
      libraryWorkflowId = getCompletedLibraryWorkflowRun(
        before,
        getProject(payload.projectId),
        payload.queueItemId,
        viewerUserId
      );
    } catch {
      // The run already routed; a failed bump only leaves the template's last run where it was.
    }

    if (!libraryWorkflowId || !isCurrent()) {
      return;
    }

    const touch = async (workflowId: string): Promise<void> => {
      const { isBundledLibraryWorkflowId, touchLibraryWorkflowLastRunAt } = await import('@features/workflow/queries');

      if (!isBundledLibraryWorkflowId(workflowId) && isCurrent()) {
        await touchLibraryWorkflowLastRunAt(workflowId, owner.signal);
      }
    };

    void touch(libraryWorkflowId).catch(() => {});
  };
