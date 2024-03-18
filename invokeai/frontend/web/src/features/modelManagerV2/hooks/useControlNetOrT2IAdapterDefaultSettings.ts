import { skipToken } from '@reduxjs/toolkit/query';
import { isNil } from 'lodash-es';
import { useMemo } from 'react';
import { useGetModelConfigWithTypeGuard } from 'services/api/hooks/useGetModelConfigWithTypeGuard';
import { isControlNetOrT2IAdapterModelConfig } from 'services/api/types';

export const useControlNetOrT2IAdapterDefaultSettings = (modelKey?: string | null) => {
  const { modelConfig, isLoading } = useGetModelConfigWithTypeGuard(
    modelKey ?? skipToken,
    isControlNetOrT2IAdapterModelConfig
  );

  const defaultSettingsDefaults = useMemo(() => {
    return {
      preprocessor: {
        isEnabled: !isNil(modelConfig?.default_settings?.preprocessor),
        value: modelConfig?.default_settings?.preprocessor || 'none',
      },
    };
  }, [modelConfig?.default_settings]);

  return { defaultSettingsDefaults, isLoading };
};
