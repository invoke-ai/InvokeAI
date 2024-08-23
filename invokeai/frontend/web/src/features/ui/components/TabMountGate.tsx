import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';

export const TabMountGate = memo(({ tab, children }: PropsWithChildren<{ tab: TabName }>) => {
  const selectIsTabEnabled = useMemo(
    () => createSelector(selectConfigSlice, (config) => !config.disabledTabs.includes(tab)),
    [tab]
  );
  const isEnabled = useAppSelector(selectIsTabEnabled);

  if (!isEnabled) {
    return null;
  }

  return children;
});

TabMountGate.displayName = 'TabMountGate';
