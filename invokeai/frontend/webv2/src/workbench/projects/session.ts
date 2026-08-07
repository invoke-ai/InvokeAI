import type { LaunchpadIntentId } from '@workbench/launchpad/intents';
import type { AccountState, WorkbenchState } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { getClientStateValue, setClientStateValue } from './api';

/**
 * The per-user session blob in the client-state KV: the editor session — which
 * projects are open as tabs and which is active — plus a legacy account snapshot
 * for older installs. Current settings live in `settings.ts` so Home can load
 * them without mounting the workbench provider.
 *
 * `openProjectIds` arrived with the library/session split. Blobs written
 * before it have no open set; `undefined` there means "unknown — open every
 * project", which is exactly what those versions did, so old sessions migrate
 * without anything visibly changing.
 */

export const SESSION_STATE_KEY = 'webv2:workbench-account';

/**
 * Search params understood by the /app route: `project` deep-links a library
 * project into the session; `new` opens the editor with a fresh draft, and
 * `intent` says how that draft should be arranged (see `launchpad/intents`).
 */
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
      openProjectIds: Array.isArray(parsed.openProjectIds)
        ? parsed.openProjectIds.filter((id): id is string => typeof id === 'string')
        : undefined,
    };
  } catch {
    return null;
  }
};

/** The open set is derived from workbench state: open tabs are the session. */
export const serializeSessionBlob = (state: WorkbenchState): string =>
  JSON.stringify({
    account: state.account,
    activeProjectId: state.activeProjectId,
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

/**
 * Take a deleted project out of the saved session.
 *
 * A project is deleted from surfaces that may not have the editor mounted, and the session blob is
 * what the `/app` guard and the Launchpad's "open" grouping read. Leaving the id there means the
 * next boot tries to hydrate a project the server no longer has, and the Launchpad shows it as
 * open until something else rewrites the blob.
 *
 * Best-effort and silent: the deletion has already happened, and failing to tidy the session after
 * it is not a reason to tell someone their project was not deleted.
 */
export const pruneSessionProject = async (projectId: string, signal?: AbortSignal): Promise<void> => {
  try {
    const blob = await fetchSessionBlob(signal);

    if (!blob?.openProjectIds?.includes(projectId)) {
      return;
    }

    const openProjectIds = blob.openProjectIds.filter((id) => id !== projectId);

    await setClientStateValue(
      SESSION_STATE_KEY,
      JSON.stringify({
        account: blob.account,
        activeProjectId: blob.activeProjectId === projectId ? (openProjectIds[0] ?? '') : blob.activeProjectId,
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
export const peekOpenProjectIds = async (): Promise<string[] | null> => {
  const blob = await fetchSessionBlob();

  return blob?.openProjectIds ?? null;
};
