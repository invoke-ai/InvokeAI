import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { useEffect, useMemo } from 'react';

import { panelRegistry } from './panelApiRegistry';

/**
 * Hook that initializes the global panel registry with the Redux store.
 */
export const usePanelRegistryInit = () => {
  useAssertSingleton('usePanelRegistryInit');
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
    panelRegistry.setTabApi(tabApi);
  }, [store, tabApi]);
};
