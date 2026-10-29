import type { ProjectSummary } from '@workbench/projects/library';

import { getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';

import { prunePinnedProjectIds, toggleProjectPin } from './projectLibraryView';

/** Centralize account pin writes and read the live snapshot so rapid toggles do not overwrite each other. */

export const toggleProjectPinPreference = (projectId: string): void => {
  const current = getWorkbenchPreferences().launchpadPinnedProjectIds;

  void patchWorkbenchPreferences({ launchpadPinnedProjectIds: toggleProjectPin(current, projectId) });
};

export const dropProjectPin = (projectId: string): void => {
  const current = getWorkbenchPreferences().launchpadPinnedProjectIds;

  if (!current.includes(projectId)) {
    return;
  }

  void patchWorkbenchPreferences({
    launchpadPinnedProjectIds: current.filter((id) => id !== projectId),
  });
};

/**
 * Drop pins whose project no longer exists — deleted from another device or
 * another tab. Writes only when something actually changed, so this is safe to
 * call after every library refresh.
 */
export const prunePinnedProjects = (summaries: readonly ProjectSummary[]): void => {
  const current = getWorkbenchPreferences().launchpadPinnedProjectIds;

  if (current.length === 0) {
    return;
  }

  const pruned = prunePinnedProjectIds(current, summaries);

  if (pruned.length !== current.length) {
    void patchWorkbenchPreferences({ launchpadPinnedProjectIds: pruned });
  }
};
