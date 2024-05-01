import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { caLayerModelChanged, selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useControlNetAndT2IAdapterModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig, ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

type Props = {
  layerId: string;
};

export const CALayerModelCombobox = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const caModelKey = useAppSelector((s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter.model?.key);
  const currentBaseModel = useAppSelector((s) => s.generation.model?.base);

  const [modelConfigs, { isLoading }] = useControlNetAndT2IAdapterModels();
  const selectedModel = useMemo(() => modelConfigs.find((m) => m.key === caModelKey), [modelConfigs, caModelKey]);

  const _onChange = useCallback(
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null) => {
      if (!modelConfig) {
        return;
      }
      dispatch(
        caLayerModelChanged({
          layerId,
          modelConfig,
        })
      );
    },
    [dispatch, layerId]
  );

  const getIsDisabled = useCallback(
    (model: AnyModelConfig): boolean => {
      const isCompatible = currentBaseModel === model.base;
      const hasMainModel = Boolean(currentBaseModel);
      return !hasMainModel || !isCompatible;
    },
    [currentBaseModel]
  );

  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel,
    getIsDisabled,
    isLoading,
  });

  return (
    <Tooltip label={selectedModel?.description}>
      <FormControl isInvalid={!value || currentBaseModel !== selectedModel?.base} w="full">
        <Combobox
          options={options}
          placeholder={t('controlnet.selectModel')}
          value={value}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    </Tooltip>
  );
});

CALayerModelCombobox.displayName = 'CALayerModelCombobox';
