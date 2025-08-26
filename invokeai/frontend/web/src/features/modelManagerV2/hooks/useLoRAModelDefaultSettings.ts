import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { isNil } from 'es-toolkit/compat';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { useMemo } from 'react';
import type { LoRAModelConfig } from 'services/api/types';

const initialStatesSelector = createMemoizedSelector(selectConfigSlice, (config) => {
  const { weight } = config.lora;

  return {
    initialWeight: weight.initial,
  };
});

export const useLoRAModelDefaultSettings = (modelConfig: LoRAModelConfig) => {
  const { initialWeight } = useAppSelector(initialStatesSelector);

  const defaultSettingsDefaults = useMemo(() => {
    return {
      weight: {
        isEnabled: !isNil(modelConfig?.default_settings?.weight),
        value: modelConfig?.default_settings?.weight || initialWeight,
      },
    };
  }, [modelConfig?.default_settings, initialWeight]);

  return defaultSettingsDefaults;
};
