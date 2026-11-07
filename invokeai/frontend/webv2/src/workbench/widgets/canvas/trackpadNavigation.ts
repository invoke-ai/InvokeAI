/**
 * Trackpad navigation turns itself on, silently, at the first pinch the canvas detects, unless the user already decided
 * it. Scrolling cannot tell a trackpad from a mouse, but a pinch only comes from a trackpad. It turns on as the user's
 * default, since the device belongs to the user rather than a project; a project's own pin still wins.
 * `canvasTrackpadNavigationDecided` records the decision for the account, so an explicit off stays off; the check reads
 * the saved preferences, so a window with a stale snapshot cannot override a choice made in another.
 */

import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import {
  areWorkbenchPreferencesHydrated,
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  updateSavedWorkbenchPreferences,
} from '@workbench/settings/store';

import { CANVAS_TRACKPAD_NAVIGATION_KEY } from './canvasSettings';

/** Whether a pinch may still turn the setting on, by this window's snapshot; the write re-checks the saved state. */
export const isTrackpadNavigationUndecided = (): boolean =>
  areWorkbenchPreferencesHydrated() && !getWorkbenchPreferences().canvasTrackpadNavigationDecided;

/** Records that the user set trackpad navigation themselves; no later pinch changes it. */
export const recordTrackpadNavigationChoice = (): void => {
  if (isTrackpadNavigationUndecided()) {
    void patchWorkbenchPreferences({ canvasTrackpadNavigationDecided: true });
  }
};

/** The flag and the one default key, over `saved`; null once the setting was decided. */
const enableOver = (saved: WorkbenchPreferences): Partial<WorkbenchPreferences> | null =>
  saved.canvasTrackpadNavigationDecided
    ? null
    : {
        canvasTrackpadNavigationDecided: true,
        scopedDefaults: {
          ...saved.scopedDefaults,
          canvas: { ...saved.scopedDefaults.canvas, [CANVAS_TRACKPAD_NAVIGATION_KEY]: true },
        },
      };

/** Turns trackpad navigation on as the user's default unless the saved preferences say it was decided. */
export const autoEnableTrackpadNavigation = (): Promise<boolean> => updateSavedWorkbenchPreferences(enableOver);
