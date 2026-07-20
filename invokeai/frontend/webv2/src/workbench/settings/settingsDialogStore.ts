import type { SettingsSectionId } from '@workbench/widgetContracts';

import { createExternalStore } from '@platform/state/externalStore';

/**
 * Open/close state for the workbench settings dialog, addressable from
 * anywhere: widget frames, menus, and commands call `openWorkbenchSettings`
 * with the section they want. The dialog itself is hosted by the top bar's
 * `SettingsButton`, which subscribes to this store.
 */

interface SettingsDialogSnapshot {
  isOpen: boolean;
  sectionId: SettingsSectionId;
}

export const settingsDialogStore = createExternalStore<SettingsDialogSnapshot>({
  isOpen: false,
  sectionId: 'appearance',
});

/** Open the workbench settings dialog, optionally at a specific section. */
export const openWorkbenchSettings = (sectionId: SettingsSectionId = 'appearance'): void => {
  settingsDialogStore.setSnapshot({ isOpen: true, sectionId });
};

export const closeWorkbenchSettings = (): void => {
  settingsDialogStore.patchSnapshot({ isOpen: false });
};

export const setWorkbenchSettingsSection = (sectionId: SettingsSectionId): void => {
  settingsDialogStore.patchSnapshot({ sectionId });
};
