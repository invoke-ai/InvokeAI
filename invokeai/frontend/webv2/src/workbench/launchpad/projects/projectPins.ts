import type { ProjectSummary } from '@workbench/projects/library';

import { getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';

import { prunePinnedProjectIds, toggleProjectPin } from './projectLibraryView';

/**
 * Pin writes, in one place.
 *
 * Pins live in account preferences, so they follow the user across devices —
 * which also means a stale id persists forever unless something removes it.
 * Home and the library both pin, and both now go through here rather than
 * each reimplementing the read-modify-write.
 *
 * Every write reads the live snapshot rather than a captured render value, so
 * two quick toggles cannot drop the first one.
 */

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
