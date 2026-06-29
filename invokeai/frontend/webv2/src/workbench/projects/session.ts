import type { AccountState, WorkbenchPreferences, WorkbenchState } from '@workbench/types';

import { getClientStateValue } from './api';

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
 * project into the session; `new` opens the editor with a fresh draft.
 */
export interface WorkbenchSearch {
  new?: true;
  project?: string;
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

export const fetchSessionBlob = async (): Promise<WorkbenchSessionBlob | null> => {
  try {
    return parseSessionBlob(await getClientStateValue(SESSION_STATE_KEY));
  } catch {
    return null;
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
