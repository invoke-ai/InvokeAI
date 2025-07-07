import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { 
  selectActiveTab, 
  selectActiveTabCanvasMainPanel,
  selectActiveTabGenerateMainPanel,
  selectActiveTabUpscalingMainPanel,
  selectActiveTabWorkflowsMainPanel
} from 'features/ui/store/uiSelectors';
import { 
  activeTabCanvasMainPanelChanged,
  activeTabGenerateMainPanelChanged,
  activeTabUpscalingMainPanelChanged,
  activeTabWorkflowsMainPanelChanged,
  dockviewPanelStateChanged,
  gridviewPanelStateChanged, 
  setActiveTab} from 'features/ui/store/uiSlice';
import type { 
  CanvasMainPanelTabName,
  DockviewPanelState, 
  GenerateMainPanelTabName,
  GridviewPanelState, 
  TabName, 
  UpscalingMainPanelTabName,
  WorkflowsMainPanelTabName
} from 'features/ui/store/uiTypes';
import { useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';

/**
 * Hook that initializes the global navigation API with callbacks to access and modify the active tab
 * and manage panel state persistence.
 */
export const useNavigationApi = () => {
  useAssertSingleton('useNavigationApi');
  const store = useAppStore();

  const getAppTab = useCallback(() => {
    return selectActiveTab(store.getState());
  }, [store]);

  const setAppTab = useCallback(
    (tab: TabName) => {
      store.dispatch(setActiveTab(tab));
    },
    [store]
  );

  const getGridviewPanelState = useCallback(
    (id: string): GridviewPanelState | undefined => {
      return store.getState().ui.gridviewPanelStates[id];
    },
    [store]
  );

  const setGridviewPanelState = useCallback(
    (id: string, state: GridviewPanelState) => {
      store.dispatch(gridviewPanelStateChanged({ id, panelState: state }));
    },
    [store]
  );

  const getDockviewPanelState = useCallback(
    (id: string): DockviewPanelState | undefined => {
      return store.getState().ui.dockviewPanelStates[id];
    },
    [store]
  );

  const setDockviewPanelState = useCallback(
    (id: string, state: DockviewPanelState) => {
      store.dispatch(dockviewPanelStateChanged({ id, panelState: state }));
    },
    [store]
  );

  const getActiveTabMainPanel = useCallback(
    (tab: TabName): string | undefined => {
      switch (tab) {
        case 'canvas':
          return selectActiveTabCanvasMainPanel(store.getState());
        case 'generate':
          return selectActiveTabGenerateMainPanel(store.getState());
        case 'upscaling':
          return selectActiveTabUpscalingMainPanel(store.getState());
        case 'workflows':
          return selectActiveTabWorkflowsMainPanel(store.getState());
        default:
          return undefined;
      }
    },
    [store]
  );

  const setActiveTabMainPanel = useCallback(
    (tab: TabName, panel: string) => {
      switch (tab) {
        case 'canvas':
          store.dispatch(activeTabCanvasMainPanelChanged(panel as CanvasMainPanelTabName));
          break;
        case 'generate':
          store.dispatch(activeTabGenerateMainPanelChanged(panel as GenerateMainPanelTabName));
          break;
        case 'upscaling':
          store.dispatch(activeTabUpscalingMainPanelChanged(panel as UpscalingMainPanelTabName));
          break;
        case 'workflows':
          store.dispatch(activeTabWorkflowsMainPanelChanged(panel as WorkflowsMainPanelTabName));
          break;
      }
    },
    [store]
  );

  useEffect(() => {
    navigationApi.connectToApp({
      getAppTab,
      setAppTab,
      panelStateCallbacks: {
        getGridviewPanelState,
        setGridviewPanelState,
        getDockviewPanelState,
        setDockviewPanelState,
        getActiveTabMainPanel,
        setActiveTabMainPanel,
      }
    });
  }, [getAppTab, setAppTab, getGridviewPanelState, setGridviewPanelState, getDockviewPanelState, setDockviewPanelState, getActiveTabMainPanel, setActiveTabMainPanel, store]);
};
