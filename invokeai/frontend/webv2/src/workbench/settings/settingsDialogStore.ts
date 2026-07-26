import type { SettingsSectionId } from '@workbench/widgetContracts';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
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

const INITIAL_SETTINGS_DIALOG_SNAPSHOT: SettingsDialogSnapshot = {
  isOpen: false,
  sectionId: 'appearance',
};

export const settingsDialogStore = createExternalStore<SettingsDialogSnapshot>(INITIAL_SETTINGS_DIALOG_SNAPSHOT);

registerAccountOwnedResource({
  clear: () => {
    settingsDialogStore.setSnapshot(INITIAL_SETTINGS_DIALOG_SNAPSHOT);
  },
  name: 'settings-dialog',
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
