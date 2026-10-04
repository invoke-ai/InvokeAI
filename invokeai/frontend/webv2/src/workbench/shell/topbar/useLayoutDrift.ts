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

// Keep selector identities stable because drift comparison serializes the entire arrangement.
const selectActiveLayoutPresetId = (snapshot: WorkbenchSnapshot): LayoutPresetId =>
  snapshot.activeProject.layout.presetId;

const selectHasDrifted = (snapshot: WorkbenchSnapshot): boolean => {
  const activePreset = resolveSavedLayoutPreset(snapshot.account, snapshot.activeProject.layout.presetId);

  return !areLayoutPresetSnapshotsEqual(createLayoutPresetSnapshot(snapshot.activeProject), activePreset.snapshot);
};

/** Read active preset immediately so the pressed tab acknowledges before debounced drift settles. */
export const useActiveLayoutPresetId = (): LayoutPresetId =>
  useWorkbenchSelector(selectActiveLayoutPresetId, Object.is);

/** Debounce drift across multi-dispatch gestures so only the settled arrangement changes the indicator. */
export const useLayoutDrift = (): LayoutDriftState => ({
  hasDrifted: useDebouncedWorkbenchSelector(selectHasDrifted, DRIFT_SETTLE_MS),
});
