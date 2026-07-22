import { applyThemeToRoot } from '@theme/applyTheme';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';

import type { SettingsEntryDeps } from './entries';

/**
 * Live workbench bindings for the palette's settings entries, shared by both
 * palette hosts. Theme preview writes the DOM directly and never persists;
 * clearing re-applies whatever the store holds at that moment, so a preview
 * followed by an actual pick reverts to the picked value (a no-op).
 */
export const SETTINGS_ENTRY_DEPS: SettingsEntryDeps = {
  clearThemePreview: () => applyThemeToRoot(getWorkbenchPreferences().themeId),
  openSettingsSection: openWorkbenchSettings,
  patchPreferences: patchWorkbenchPreferences,
  previewTheme: applyThemeToRoot,
};
