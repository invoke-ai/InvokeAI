import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useOptionalWorkbenchSelector } from '@workbench/WorkbenchContext';

import type { CustomizeSession, StructureLock } from './structureLock';

import { getCustomizeSession, getStructureLock } from './structureLock';

/** The active project's Customize session, draft included, or null; only the Customize bar needs its contents. */
export const useCustomizeSession = (): CustomizeSession | null =>
  useOptionalWorkbenchSelector(
    ({ activeProject, customizing }) => getCustomizeSession(customizing, activeProject),
    null,
    Object.is
  );

/** Whether the active project customizes its workspace; false outside an editor. A draft edit re-renders nothing. */
export const useIsCustomizing = (): boolean =>
  useOptionalWorkbenchSelector(
    ({ activeProject, customizing }) => getCustomizeSession(customizing, activeProject) !== null,
    false,
    Object.is
  );

/** The shell's one read of the lock: its structural chrome renders only while this says structure may change. */
export const useStructureLock = (): StructureLock => {
  const rearrangeWithoutCustomize = useWorkbenchPreferenceSelector(
    (preferences) => preferences.rearrangeWithoutCustomize
  );

  return getStructureLock(useIsCustomizing(), rearrangeWithoutCustomize);
};

export const useCanEditStructure = (): boolean => useStructureLock().canEditStructure;
