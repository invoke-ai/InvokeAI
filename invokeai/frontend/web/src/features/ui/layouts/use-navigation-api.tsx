import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { useCallback, useEffect, useMemo } from 'react';

import { navigationApi } from './navigation-api';
import { navigationApi as navigationApi2 } from './navigation-api-2';

/**
 * Hook that initializes the global navigation API with callbacks to access and modify the active tab.
 */
export const useNavigationApi = () => {
  useAssertSingleton('useNavigationApi');
  const store = useAppStore();
  const tabApi = useMemo(
    () => ({
      getTab: () => {
        return selectActiveTab(store.getState());
      },
      setTab: (tab: TabName) => {
        store.dispatch(setActiveTab(tab));
      },
    }),
    [store]
  );

  useEffect(() => {
    navigationApi.setTabApi(tabApi);
  }, [store, tabApi]);

  const getAppTab = useCallback(() => {
    return selectActiveTab(store.getState());
  }, [store]);
  const setAppTab = useCallback(
    (tab: TabName) => {
      store.dispatch(setActiveTab(tab));
    },
    [store]
  );
  useEffect(() => {
    navigationApi2.connectToApp({ getAppTab, setAppTab });
  }, [getAppTab, setAppTab, store, tabApi]);
};
