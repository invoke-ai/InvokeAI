import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import type { AppFeature, SDFeature } from 'app/types/invokeai';
import type { TabName } from 'features/controlLayers/store/types';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { useMemo } from 'react';

export const useFeatureStatus = (feature: AppFeature | SDFeature | TabName) => {
  const selectIsFeatureEnabled = useMemo(
    () =>
      createSelector(selectConfigSlice, (config) => {
        return !(
          config.disabledFeatures.includes(feature as AppFeature) ||
          config.disabledSDFeatures.includes(feature as SDFeature) ||
          config.disabledTabs.includes(feature as TabName)
        );
      }),
    [feature]
  );

  const isFeatureEnabled = useAppSelector(selectIsFeatureEnabled);

  return isFeatureEnabled;
};
