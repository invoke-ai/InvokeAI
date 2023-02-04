import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import _ from 'lodash';
import { tabMap } from './tabMap';
import { UIState } from './uiTypes';

export const activeTabNameSelector = createSelector(
  (state: RootState) => state.ui,
  (ui: UIState) => tabMap[ui.activeTab],
  {
    memoizeOptions: {
      equalityCheck: _.isEqual,
    },
  }
);

export const activeTabIndexSelector = createSelector(
  (state: RootState) => state.ui,
  (ui: UIState) => ui.activeTab,
  {
    memoizeOptions: {
      equalityCheck: _.isEqual,
    },
  }
);

export const uiSelector = createSelector(
  (state: RootState) => state.ui,
  (ui) => ui,
  {
    memoizeOptions: {
      equalityCheck: _.isEqual,
    },
  }
);
