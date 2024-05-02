import { createSelector } from '@reduxjs/toolkit';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { selectUiSlice } from 'features/ui/store/uiSlice';
import { isString } from 'lodash-es';

import { TAB_NUMBER_MAP } from './tabMap';

export const activeTabNameSelector = createSelector(
  selectUiSlice,
  /**
   * Previously `activeTab` was an integer, but now it's a string.
   * Default to first tab in case user has integer.
   */
  (ui) => (isString(ui.activeTab) ? ui.activeTab : 'generation')
);

export const activeTabIndexSelector = createSelector(selectUiSlice, selectConfigSlice, (ui, config) => {
  const tabs = TAB_NUMBER_MAP.filter((t) => !config.disabledTabs.includes(t));
  const idx = tabs.indexOf(ui.activeTab);
  return idx === -1 ? 0 : idx;
});
