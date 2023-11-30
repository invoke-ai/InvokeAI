import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { isEqual, isString } from 'lodash-es';
import { tabMap } from './tabMap';

export const activeTabNameSelector = createSelector(
  (state: RootState) => state,
  /**
   * Previously `activeTab` was an integer, but now it's a string.
   * Default to first tab in case user has integer.
   */
  ({ ui }) => (isString(ui.activeTab) ? ui.activeTab : 'txt2img')
);

export const activeTabIndexSelector = createSelector(
  (state: RootState) => state,
  ({ ui, config }) => {
    const tabs = tabMap.filter((t) => !config.disabledTabs.includes(t));
    const idx = tabs.indexOf(ui.activeTab);
    return idx === -1 ? 0 : idx;
  }
);

export const uiSelector = createSelector(
  (state: RootState) => state.ui,
  (ui) => ui,
  {
    memoizeOptions: {
      equalityCheck: isEqual,
    },
  }
);
