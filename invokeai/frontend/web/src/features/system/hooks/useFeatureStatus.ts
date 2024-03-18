import { useAppSelector } from 'app/store/storeHooks';
import type { AppFeature, SDFeature } from 'app/types/invokeai';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { useMemo } from 'react';

export const useFeatureStatus = (feature: AppFeature | SDFeature | InvokeTabName) => {
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);

  const disabledFeatures = useAppSelector((s) => s.config.disabledFeatures);

  const disabledSDFeatures = useAppSelector((s) => s.config.disabledSDFeatures);

  const isFeatureDisabled = useMemo(
    () =>
      disabledFeatures.includes(feature as AppFeature) ||
      disabledSDFeatures.includes(feature as SDFeature) ||
      disabledTabs.includes(feature as InvokeTabName),
    [disabledFeatures, disabledSDFeatures, disabledTabs, feature]
  );

  const isFeatureEnabled = useMemo(
    () =>
      !(
        disabledFeatures.includes(feature as AppFeature) ||
        disabledSDFeatures.includes(feature as SDFeature) ||
        disabledTabs.includes(feature as InvokeTabName)
      ),
    [disabledFeatures, disabledSDFeatures, disabledTabs, feature]
  );

  return { isFeatureDisabled, isFeatureEnabled };
};
