import type { LayoutPresetId } from '@workbench/layoutContracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import {
  areLayoutPresetSnapshotsEqual,
  createLayoutPresetSnapshot,
  resolveSavedLayoutPreset,
} from '@workbench/layoutPresetSnapshots';
import { useDebouncedWorkbenchSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';

/** How long the live layout must hold still before the drift dot reacts. */
const DRIFT_SETTLE_MS = 250;

export interface LayoutDriftState {
  hasDrifted: boolean;
}

// Module scope, not inline closures: a selector whose identity changed every
// render would defeat `useSyncExternalStoreWithSelector`'s memoization and
// re-run it on each one — and the drift comparison serialises the whole
// arrangement.
const selectActiveLayoutPresetId = (snapshot: WorkbenchSnapshot): LayoutPresetId =>
  snapshot.activeProject.layout.presetId;

const selectHasDrifted = (snapshot: WorkbenchSnapshot): boolean => {
  const activePreset = resolveSavedLayoutPreset(snapshot.account, snapshot.activeProject.layout.presetId);

  return !areLayoutPresetSnapshotsEqual(createLayoutPresetSnapshot(snapshot.activeProject), activePreset.snapshot);
};

/**
 * Which preset the active project is on, read live.
 *
 * Deliberately not routed through the drift debounce below: the strip renders
 * the selected tab from this, and a debounced read meant the control the user
 * pressed was the last thing on screen to acknowledge the press.
 */
export const useActiveLayoutPresetId = (): LayoutPresetId =>
  useWorkbenchSelector(selectActiveLayoutPresetId, Object.is);

/**
 * Whether the live dock layout has diverged from the preset it was loaded from.
 *
 * Debounced rather than read live: a drop that lands a widget, resizes its
 * region, and reveals it arrives as several dispatches in a row, and a dot that
 * blinks part-way through a gesture reads as a glitch rather than as
 * information. Settling first means the user only ever sees the resting answer.
 */
export const useLayoutDrift = (): LayoutDriftState => ({
  hasDrifted: useDebouncedWorkbenchSelector(selectHasDrifted, DRIFT_SETTLE_MS),
});
