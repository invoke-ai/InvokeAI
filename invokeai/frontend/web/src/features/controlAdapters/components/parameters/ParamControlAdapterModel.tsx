import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { useControlAdapterCLIPVisionModel } from 'features/controlAdapters/hooks/useControlAdapterCLIPVisionModel';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterModel } from 'features/controlAdapters/hooks/useControlAdapterModel';
import { useControlAdapterModels } from 'features/controlAdapters/hooks/useControlAdapterModels';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import {
  controlAdapterCLIPVisionModelChanged,
  controlAdapterModelChanged,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import type { CLIPVisionModel } from 'features/controlAdapters/store/types';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  AnyModelConfig,
  ControlNetModelConfig,
  IPAdapterModelConfig,
  T2IAdapterModelConfig,
} from 'services/api/types';

type ParamControlAdapterModelProps = {
  id: string;
};

const selectMainModel = createMemoizedSelector(selectGenerationSlice, (generation) => generation.model);

const ParamControlAdapterModel = ({ id }: ParamControlAdapterModelProps) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const controlAdapterType = useControlAdapterType(id);
  const { modelConfig } = useControlAdapterModel(id);
  const dispatch = useAppDispatch();
  const currentBaseModel = useAppSelector((s) => s.generation.model?.base);
  const currentCLIPVisionModel = useControlAdapterCLIPVisionModel(id);
  const mainModel = useAppSelector(selectMainModel);
  const { t } = useTranslation();

  const [modelConfigs, { isLoading }] = useControlAdapterModels(controlAdapterType);

  const _onChange = useCallback(
    (modelConfig: ControlNetModelConfig | IPAdapterModelConfig | T2IAdapterModelConfig | null) => {
      if (!modelConfig) {
        return;
      }
      dispatch(
        controlAdapterModelChanged({
          id,
          modelConfig,
        })
      );
    },
    [dispatch, id]
  );

  const onCLIPVisionModelChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v?.value) {
        return;
      }
      dispatch(controlAdapterCLIPVisionModelChanged({ id, clipVisionModel: v.value as CLIPVisionModel }));
    },
    [dispatch, id]
  );

  const selectedModel = useMemo(
    () => (modelConfig && controlAdapterType ? { ...modelConfig, model_type: controlAdapterType } : null),
    [controlAdapterType, modelConfig]
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

  const clipVisionOptions = useMemo<ComboboxOption[]>(
    () => [
      { label: 'ViT-H', value: 'ViT-H' },
      { label: 'ViT-G', value: 'ViT-G' },
    ],
    []
  );

  const clipVisionModel = useMemo(
    () => clipVisionOptions.find((o) => o.value === currentCLIPVisionModel),
    [clipVisionOptions, currentCLIPVisionModel]
  );

  return (
    <Flex sx={{ gap: 2 }}>
      <Tooltip label={value?.description}>
        <FormControl
          isDisabled={!isEnabled}
          isInvalid={!value || mainModel?.base !== modelConfig?.base}
          sx={{ width: '100%' }}
        >
          <Combobox
            options={options}
            placeholder={t('controlnet.selectModel')}
            value={value}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
          />
        </FormControl>
      </Tooltip>
      {modelConfig?.type === 'ip_adapter' && modelConfig.format === 'checkpoint' && (
        <FormControl
          isDisabled={!isEnabled}
          isInvalid={!value || mainModel?.base !== modelConfig?.base}
          sx={{ width: 'max-content', minWidth: 28 }}
        >
          <Combobox
            options={clipVisionOptions}
            placeholder={t('controlnet.selectCLIPVisionModel')}
            value={clipVisionModel}
            onChange={onCLIPVisionModelChange}
          />
        </FormControl>
      )}
    </Flex>
  );
};

export default memo(ParamControlAdapterModel);
