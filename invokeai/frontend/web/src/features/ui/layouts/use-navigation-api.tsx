import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';

/**
 * Hook that initializes the global navigation API with callbacks to access and modify the active tab.
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
  useEffect(() => {
    navigationApi.connectToApp({ getAppTab, setAppTab });
  }, [getAppTab, setAppTab, store]);
};
