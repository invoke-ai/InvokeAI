import { WORKBENCH_LANGUAGE_OPTIONS } from '@platform/i18n/languages';
import { applyThemeToRoot } from '@theme/applyTheme';
import { THEMES } from '@theme/themes';
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
  languageOptions: WORKBENCH_LANGUAGE_OPTIONS,
  openSettingsSection: openWorkbenchSettings,
  patchPreferences: patchWorkbenchPreferences,
  previewTheme: applyThemeToRoot,
  themes: THEMES,
};
