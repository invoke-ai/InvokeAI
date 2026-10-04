import type { LaunchpadIntentId } from '@workbench/launchpad/intents';
import type { AccountState, WorkbenchState } from '@workbench/projectContracts';
import type { QueueRunJournal } from '@workbench/queue-integration/queueRunJournal';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import type { ProjectDraftStore } from './draftStore';

import { getClientStateValue, setClientStateValue } from './api';

/** Session tabs are separate from settings; legacy undefined openProjectIds means unknown/open-all. */

export const SESSION_STATE_KEY = 'webv2:workbench-account';

/** project opens a library project; new creates a draft whose intent picks the arrangement and source. */
export interface WorkbenchSearch {
  new?: true;
  project?: string;
  intent?: LaunchpadIntentId;
}

export interface WorkbenchSessionBlob {
  account: AccountState & {
    /** Written by builds that kept preferences in the account; read once as a migration source. */
    preferences?: Partial<WorkbenchPreferences>;
  };
  activeProjectId: string;
  draftEditorSessionIds?: Record<string, string>;
  editorSessionId?: string;
  openProjectIds?: string[];
}

export const parseSessionBlob = (raw: string | null): WorkbenchSessionBlob | null => {
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<WorkbenchSessionBlob>;

    if (!parsed.account || typeof parsed.activeProjectId !== 'string') {
      return null;
    }

    return {
      account: parsed.account,
      activeProjectId: parsed.activeProjectId,
      draftEditorSessionIds:
        parsed.draftEditorSessionIds && typeof parsed.draftEditorSessionIds === 'object'
          ? Object.fromEntries(
              Object.entries(parsed.draftEditorSessionIds).filter(
                (entry): entry is [string, string] => entry[0].length > 0 && typeof entry[1] === 'string'
              )
            )
          : undefined,
      editorSessionId: typeof parsed.editorSessionId === 'string' ? parsed.editorSessionId : undefined,
      openProjectIds: Array.isArray(parsed.openProjectIds)
        ? parsed.openProjectIds.filter((id): id is string => typeof id === 'string')
        : undefined,
    };
  } catch {
    return null;
  }
};

/** The open set is derived from workbench state: open tabs are the session. */
export const serializeSessionBlob = (
  state: WorkbenchState,
  editorSessionId?: string,
  draftEditorSessionIds?: Record<string, string>
): string =>
  JSON.stringify({
    account: state.account,
    activeProjectId: state.activeProjectId,
    ...(draftEditorSessionIds && Object.keys(draftEditorSessionIds).length > 0 ? { draftEditorSessionIds } : {}),
    ...(editorSessionId ? { editorSessionId } : {}),
    openProjectIds: state.projects.map((project) => project.id),
  } satisfies WorkbenchSessionBlob);

export const fetchSessionBlob = async (signal?: AbortSignal): Promise<WorkbenchSessionBlob | null> => {
  try {
    return parseSessionBlob(await getClientStateValue(SESSION_STATE_KEY, signal));
  } catch {
    signal?.throwIfAborted();

    return null;
  }
};

export const fetchSessionBlobStrict = async (signal?: AbortSignal): Promise<WorkbenchSessionBlob | null> =>
  parseSessionBlob(await getClientStateValue(SESSION_STATE_KEY, signal));

type DurableRecoveryProjectIdsResult = { kind: 'available'; projectIds: string[] } | { kind: 'unavailable' };

const listDurableRecoveryProjectIds = async (): Promise<DurableRecoveryProjectIdsResult> => {
  let draftStore: ProjectDraftStore;
  let queueRunJournal: QueueRunJournal;
  try {
    const [{ captureAccountScope }, { createAccountOwnedProjectDraftStore }, { createAccountOwnedQueueRunJournal }] =
      await Promise.all([
        import('@platform/state/accountLifecycle'),
        import('./indexedDbDraftStore'),
        import('@workbench/queue-integration/queueRunJournal'),
      ]);
    const owner = captureAccountScope();
    const [draftStoreResult, queueRunJournalResult] = await Promise.allSettled([
      createAccountOwnedProjectDraftStore(owner),
      createAccountOwnedQueueRunJournal(owner),
    ]);
    if (draftStoreResult.status === 'rejected' || queueRunJournalResult.status === 'rejected') {
      if (draftStoreResult.status === 'fulfilled') {
        draftStoreResult.value.close();
      }
      if (queueRunJournalResult.status === 'fulfilled') {
        queueRunJournalResult.value.close();
      }
      return { kind: 'unavailable' };
    }
    draftStore = draftStoreResult.value;
    queueRunJournal = queueRunJournalResult.value;
  } catch {
    return { kind: 'unavailable' };
  }

  try {
    const [drafts, retargets, queueRuns] = await Promise.all([
      draftStore.list({ limit: 1 }),
      draftStore.listRetargets({ limit: 1 }),
      queueRunJournal.listProjectIds(),
    ]);
    if (drafts.kind !== 'available' || retargets.kind !== 'available' || queueRuns.kind !== 'available') {
      return { kind: 'unavailable' };
    }
    return {
      kind: 'available',
      projectIds: [
        ...new Set([
          ...queueRuns.projectIds,
          ...drafts.items.map((draft) => draft.projectId),
          ...retargets.items.map((handoff) => handoff.targetProjectId),
        ]),
      ],
    };
  } catch {
    return { kind: 'unavailable' };
  } finally {
    draftStore.close();
    queueRunJournal.close();
  }
};

/** Session cleanup is best-effort after committed deletion and cannot change its verdict. */
export const pruneSessionProject = async (projectId: string, signal?: AbortSignal): Promise<void> => {
  try {
    const blob = await fetchSessionBlob(signal);

    if (!blob?.openProjectIds?.includes(projectId)) {
      return;
    }

    const openProjectIds = blob.openProjectIds.filter((id) => id !== projectId);
    const draftEditorSessionIds = blob.draftEditorSessionIds
      ? Object.fromEntries(Object.entries(blob.draftEditorSessionIds).filter(([id]) => id !== projectId))
      : undefined;

    await setClientStateValue(
      SESSION_STATE_KEY,
      JSON.stringify({
        account: blob.account,
        activeProjectId: blob.activeProjectId === projectId ? (openProjectIds[0] ?? '') : blob.activeProjectId,
        ...(draftEditorSessionIds && Object.keys(draftEditorSessionIds).length > 0 ? { draftEditorSessionIds } : {}),
        ...(blob.editorSessionId ? { editorSessionId: blob.editorSessionId } : {}),
        openProjectIds,
      } satisfies WorkbenchSessionBlob),
      signal
    );
  } catch {
    signal?.throwIfAborted();
  }
};

/**
 * Cheap pre-mount peek for the /app route guard: would any tabs open?
 * `null` means "could not tell" (no blob yet, a pre-split blob, or the
 * backend is unreachable) — the guard must not redirect on null, only on a
 * definite empty session.
 */
export const peekOpenProjectIds = async (
  dependencies: {
    listDurableRecoveryProjectIds?: () => Promise<DurableRecoveryProjectIdsResult>;
  } = {}
): Promise<string[] | null> => {
  const blob = await fetchSessionBlob();

  if (!blob?.openProjectIds || blob.openProjectIds.length > 0) {
    return blob?.openProjectIds ?? null;
  }

  const recovery = await (dependencies.listDurableRecoveryProjectIds ?? listDurableRecoveryProjectIds)();

  return recovery.kind === 'available' ? recovery.projectIds : null;
};
