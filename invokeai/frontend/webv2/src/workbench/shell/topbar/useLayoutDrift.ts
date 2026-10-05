import type { LayoutPresetId } from '@workbench/layoutContracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import {
  areLayoutPresetSnapshotsEqual,
  createLayoutPresetSnapshot,
  getUnsavedInactiveLayoutPresetIds,
  resolveSavedLayoutPreset,
} from '@workbench/layoutPresetSnapshots';
import { useDebouncedWorkbenchSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

/** How long the live layout must hold still before the drift dot reacts. */
const DRIFT_SETTLE_MS = 250;

export interface LayoutDriftState {
  hasDrifted: boolean;
}

/** Whose drift it is travels with it, so a value is never read as another preset's or project's. */
interface DriftSelection {
  hasDrifted: boolean;
  presetId: LayoutPresetId;
  projectId: string;
}

const isSameDrift = (left: DriftSelection, right: DriftSelection): boolean =>
  left.hasDrifted === right.hasDrifted && left.presetId === right.presetId && left.projectId === right.projectId;

// Debouncing smooths a gesture within one layout; arriving on another preset or project shows its state at once.
const isNewSubject = (previous: DriftSelection, next: DriftSelection): boolean =>
  previous.presetId !== next.presetId || previous.projectId !== next.projectId;

// Keep selector identities stable because drift comparison serializes the entire arrangement.
const selectActiveLayoutPresetId = (snapshot: WorkbenchSnapshot): LayoutPresetId =>
  snapshot.activeProject.layout.presetId;
const selectPresetWorkingLayouts = (snapshot: WorkbenchSnapshot) => snapshot.activeProject.presetWorkingLayouts;
const selectAccount = (snapshot: WorkbenchSnapshot) => snapshot.account;

const selectDrift = ({ account, activeProject }: WorkbenchSnapshot): DriftSelection => {
  const activePreset = resolveSavedLayoutPreset(account, activeProject.layout.presetId);

  return {
    hasDrifted: !areLayoutPresetSnapshotsEqual(createLayoutPresetSnapshot(activeProject), activePreset.snapshot),
    presetId: activeProject.layout.presetId,
    projectId: activeProject.id,
  };
};

/** Read active preset immediately so the pressed tab acknowledges before debounced drift settles. */
export const useActiveLayoutPresetId = (): LayoutPresetId =>
  useWorkbenchSelector(selectActiveLayoutPresetId, Object.is);

/** Debounce drift across multi-dispatch gestures so only the settled arrangement changes the indicator. */
export const useLayoutDrift = (): LayoutDriftState => ({
  hasDrifted: useDebouncedWorkbenchSelector(selectDrift, DRIFT_SETTLE_MS, isSameDrift, isNewSubject).hasDrifted,
});

/**
 * Inactive presets the active project holds unsaved arrangements of. Working copies change only when presets switch,
 * save or revert, so the comparison reruns on those and on account edits rather than on every layout change.
 */
export const useUnsavedInactiveLayoutPresetIds = (): ReadonlySet<LayoutPresetId> => {
  const presetWorkingLayouts = useWorkbenchSelector(selectPresetWorkingLayouts, Object.is);
  const account = useWorkbenchSelector(selectAccount, Object.is);
  const presetId = useActiveLayoutPresetId();

  return useMemo(
    () => new Set(getUnsavedInactiveLayoutPresetIds(presetWorkingLayouts, presetId, account)),
    [account, presetId, presetWorkingLayouts]
  );
};
