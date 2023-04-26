import { AppFeature } from 'app/invokeai';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { useMemo } from 'react';

export const useFeatureStatus = (feature: AppFeature) => {
  const disabledFeatures = useAppSelector(
    (state: RootState) => state.config.disabledFeatures
  );

  const isFeatureDisabled = useMemo(
    () => disabledFeatures.includes(feature),
    [disabledFeatures, feature]
  );

  const isFeatureEnabled = useMemo(
    () => !disabledFeatures.includes(feature),
    [disabledFeatures, feature]
  );

  return { isFeatureDisabled, isFeatureEnabled };
};
