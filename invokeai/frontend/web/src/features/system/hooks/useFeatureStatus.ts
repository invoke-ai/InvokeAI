import { AppFeature } from 'app/types/invokeai';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
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
