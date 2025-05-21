import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useCallback } from 'react';

export const useIsModelDisabled = () => {
  const isChatGPT4oHighEnabled = useFeatureStatus('chatGPT4oHigh');

  const isChatGPT4oHighModelDisabled = useCallback(
    (model: ParameterModel) => {
      return model?.base === 'chatgpt-4o' && model.name.toLowerCase().includes('high') && !isChatGPT4oHighEnabled;
    },
    [isChatGPT4oHighEnabled]
  );

  return { isChatGPT4oHighModelDisabled };
};
