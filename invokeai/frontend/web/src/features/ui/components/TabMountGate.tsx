import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';

export const TabMountGate = memo(({ tab, children }: PropsWithChildren<{ tab: InvokeTabName }>) => {
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
