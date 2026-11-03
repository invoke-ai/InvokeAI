import type { KeyboardEvent } from 'react';

import { focusSearchOnSlash } from '@platform/browser/searchShortcut';

/**
 * `/` inside the settings dialog jumps to its search field. The Preferences page gets the same key from its
 * `ManagerLayout`, which acts only while no modal is open; the dialog is one, so it handles the key itself.
 */
export const focusSettingsSearchOnSlash = (event: KeyboardEvent<HTMLElement>) => {
  focusSearchOnSlash(event.nativeEvent, event.currentTarget, '[data-manager-search]');
};
