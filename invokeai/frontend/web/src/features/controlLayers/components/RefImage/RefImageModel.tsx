import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectBase } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGlobalReferenceImageModels } from 'services/api/hooks/modelsByType';
import type {
  ChatGPT4oModelConfig,
  FLUXKontextModelConfig,
  FLUXReduxModelConfig,
  IPAdapterModelConfig,
} from 'services/api/types';

type Props = {
  modelKey: string | null;
  onChangeModel: (
    modelConfig: IPAdapterModelConfig | FLUXReduxModelConfig | ChatGPT4oModelConfig | FLUXKontextModelConfig
  ) => void;
};

export const RefImageModel = memo(({ modelKey, onChangeModel }: Props) => {
  const { t } = useTranslation();
  const currentBaseModel = useAppSelector(selectBase);
  const [modelConfigs, { isLoading }] = useGlobalReferenceImageModels();
  const selectedModel = useMemo(() => modelConfigs.find((m) => m.key === modelKey), [modelConfigs, modelKey]);

  const _onChangeModel = useCallback(
    (
      modelConfig: IPAdapterModelConfig | FLUXReduxModelConfig | ChatGPT4oModelConfig | FLUXKontextModelConfig | null
    ) => {
      if (!modelConfig) {
        return;
      }
      onChangeModel(modelConfig);
    },
    [onChangeModel]
  );

  const getIsDisabled = useCallback(
    (model: IPAdapterModelConfig | FLUXReduxModelConfig | ChatGPT4oModelConfig | FLUXKontextModelConfig): boolean => {
      const hasMainModel = Boolean(currentBaseModel);
      const hasSameBase = currentBaseModel === model.base;
      return !hasMainModel || !hasSameBase;
    },
    [currentBaseModel]
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
      <FormControl isInvalid={!value || currentBaseModel !== selectedModel?.base} w="full" minW={0}>
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
