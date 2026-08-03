import type { LayoutPreset } from '@workbench/layoutContracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import {
  areLayoutPresetSnapshotsEqual,
  createLayoutPresetSnapshot,
  resolveSavedLayoutPreset,
} from '@workbench/layoutPresetSnapshots';
import { useDebouncedWorkbenchSelector } from '@workbench/WorkbenchContext';

/** How long the live layout must hold still before the drift dot reacts. */
const DRIFT_SETTLE_MS = 250;

export interface LayoutDriftState {
  activePreset: LayoutPreset;
  hasDrifted: boolean;
}

const areDriftStatesEqual = (left: LayoutDriftState, right: LayoutDriftState): boolean =>
  left.hasDrifted === right.hasDrifted &&
  left.activePreset.id === right.activePreset.id &&
  left.activePreset.label === right.activePreset.label;

// Module scope, not an inline closure: the comparison serialises the whole
// arrangement, and a selector whose identity changed every render would defeat
// `useSyncExternalStoreWithSelector`'s memoization and re-run it on each one.
const selectLayoutDrift = (snapshot: WorkbenchSnapshot): LayoutDriftState => {
  const activePreset = resolveSavedLayoutPreset(snapshot.account, snapshot.activeProject.layout.presetId);

  return {
    activePreset,
    hasDrifted: !areLayoutPresetSnapshotsEqual(
      createLayoutPresetSnapshot(snapshot.activeProject),
      activePreset.snapshot
    ),
  };
};

/**
 * Whether the live dock layout has diverged from the preset it was loaded from.
 *
 * Debounced rather than read live: a drop that lands a widget, resizes its
 * region, and reveals it arrives as several dispatches in a row, and a dot that
 * blinks part-way through a gesture reads as a glitch rather than as
 * information. Settling first means the user only ever sees the resting answer.
 */
export const useLayoutDrift = (): LayoutDriftState =>
  useDebouncedWorkbenchSelector(selectLayoutDrift, DRIFT_SETTLE_MS, areDriftStatesEqual);
