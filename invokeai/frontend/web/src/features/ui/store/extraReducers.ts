import { InvokeTabName, tabMap } from './tabMap';
import { UIState } from './uiTypes';

export const setActiveTabReducer = (
  state: UIState,
  newActiveTab: number | InvokeTabName
) => {
  if (typeof newActiveTab === 'number') {
    state.activeTab = newActiveTab;
  } else {
    state.activeTab = tabMap.indexOf(newActiveTab);
  }
};
