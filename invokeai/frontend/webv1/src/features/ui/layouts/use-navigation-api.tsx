import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { dockviewStorageKeyChanged, setActiveTab } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { useEffect, useMemo } from 'react';
import type { JsonObject } from 'type-fest';

import { navigationApi } from './navigation-api';

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
          return store.getState().ui.panels[id];
        },
        set: (id: string, state: JsonObject) => {
          store.dispatch(dockviewStorageKeyChanged({ id, state }));
        },
        delete: (id: string) => {
          store.dispatch(dockviewStorageKeyChanged({ id, state: undefined }));
        },
      },
    }),
    [store]
  );

  useEffect(() => {
    navigationApi.connectToApp(appApi);
  }, [appApi, store]);
};
