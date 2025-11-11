import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectActiveCanvasId, selectActiveTab } from 'features/controlLayers/store/selectors';
import { setActiveTab } from 'features/controlLayers/store/tabSlice';
import type { TabName } from 'features/controlLayers/store/types';
import { dockviewStorageKeyChanged } from 'features/ui/store/uiSlice';
import { useEffect, useMemo } from 'react';
import type { JsonObject } from 'type-fest';

import { navigationApi } from './navigation-api';

const createStorageId = (store: AppStore, id: string) => {
  const state = store.getState();
  const activeTab = selectActiveTab(state);
  const activeCanvasId = selectActiveCanvasId(state);

  return activeTab === 'canvas' ? `${activeCanvasId}-${id}` : `${activeTab}-${id}`;
};

/**
 * Hook that initializes the global navigation API with callbacks to access and modify the active tab and handle
 * stored panel states.
 */
export const useNavigationApi = () => {
  useAssertSingleton('useNavigationApi');
  const store = useAppStore();

  const appApi = useMemo(
    () => ({
      activeTab: {
        get: (): TabName => {
          return selectActiveTab(store.getState());
        },
        set: (tab: TabName) => {
          store.dispatch(setActiveTab(tab));
        },
      },
      storage: {
        get: (id: string) => {
          const storageId = createStorageId(store, id);
          return store.getState().ui.panels[storageId];
        },
        set: (id: string, state: JsonObject) => {
          const storageId = createStorageId(store, id);
          store.dispatch(dockviewStorageKeyChanged({ id: storageId, state }));
        },
        delete: (id: string) => {
          const storageId = createStorageId(store, id);
          store.dispatch(dockviewStorageKeyChanged({ id: storageId, state: undefined }));
        },
      },
    }),
    [store]
  );

  useEffect(() => {
    navigationApi.connectToApp(appApi);
  }, [appApi, store]);
};
