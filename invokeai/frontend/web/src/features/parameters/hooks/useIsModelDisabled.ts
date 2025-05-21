import type { AppFeature } from 'app/types/invokeai';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useCallback } from 'react';

export const useIsModelDisabled = (feature: AppFeature) => {
  const isEnabled = useFeatureStatus(feature);

  const isModelDisabled = useCallback(
    (model: ParameterModel) => {
      return model?.base === 'chatgpt-4o' && model.name.toLowerCase().includes('high') && !isEnabled;
    },
    [isEnabled]
  );

  return isModelDisabled;
};
