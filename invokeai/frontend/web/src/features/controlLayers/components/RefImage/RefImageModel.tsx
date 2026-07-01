import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { areBasesCompatibleForRefImage } from 'features/controlLayers/store/validators';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGlobalReferenceImageModels } from 'services/api/hooks/modelsByType';
import type {
  ChatGPT4oModelConfig,
  FLUXKontextModelConfig,
  FLUXReduxModelConfig,
  Gemini2_5ModelConfig,
  IPAdapterModelConfig,
} from 'services/api/types';

type Props = {
  modelKey: string | null;
  onChangeModel: (
    modelConfig:
      | IPAdapterModelConfig
      | FLUXReduxModelConfig
      | ChatGPT4oModelConfig
      | FLUXKontextModelConfig
      | Gemini2_5ModelConfig
  ) => void;
};

export const RefImageModel = memo(({ modelKey, onChangeModel }: Props) => {
  const { t } = useTranslation();
  const mainModelConfig = useAppSelector(selectMainModelConfig);
  const [modelConfigs, { isLoading }] = useGlobalReferenceImageModels();
  const selectedModel = useMemo(() => modelConfigs.find((m) => m.key === modelKey), [modelConfigs, modelKey]);

  const _onChangeModel = useCallback(
    (
      modelConfig:
        | IPAdapterModelConfig
        | FLUXReduxModelConfig
        | ChatGPT4oModelConfig
        | FLUXKontextModelConfig
        | Gemini2_5ModelConfig
        | null
    ) => {
      if (!modelConfig) {
        return;
      }
      onChangeModel(modelConfig);
    },
    [onChangeModel]
  );

  const getIsDisabled = useCallback(
    (
      model:
        | IPAdapterModelConfig
        | FLUXReduxModelConfig
        | ChatGPT4oModelConfig
        | FLUXKontextModelConfig
        | Gemini2_5ModelConfig
    ): boolean => {
      return !areBasesCompatibleForRefImage(mainModelConfig, model);
    },
    [mainModelConfig]
  );

  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChangeModel,
    selectedModel,
    getIsDisabled,
    isLoading,
  });

  return (
    <Tooltip label={selectedModel?.description}>
      <FormControl
        isInvalid={!value || !areBasesCompatibleForRefImage(mainModelConfig, selectedModel)}
        w="full"
        minW={0}
      >
        <Combobox
          options={options}
          placeholder={t('common.placeholderSelectAModel')}
          value={value}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    </Tooltip>
  );
});

RefImageModel.displayName = 'RefImageModel';
