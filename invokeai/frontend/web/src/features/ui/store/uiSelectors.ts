import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import { isString } from 'lodash-es';

import { tabMap } from './tabMap';

export const activeTabNameSelector = createMemoizedSelector(
  (state: RootState) => state,
  /**
   * Previously `activeTab` was an integer, but now it's a string.
   * Default to first tab in case user has integer.
   */
  ({ ui }) => (isString(ui.activeTab) ? ui.activeTab : 'txt2img')
);

export const activeTabIndexSelector = createMemoizedSelector(
  (state: RootState) => state,
  ({ ui, config }) => {
    const tabs = tabMap.filter((t) => !config.disabledTabs.includes(t));
    const idx = tabs.indexOf(ui.activeTab);
    return idx === -1 ? 0 : idx;
  }
);
