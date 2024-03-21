import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterModel } from 'features/controlAdapters/hooks/useControlAdapterModel';
import { useControlAdapterModels } from 'features/controlAdapters/hooks/useControlAdapterModels';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import { controlAdapterModelChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
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

  return (
    <Tooltip label={value?.description}>
      <FormControl isDisabled={!isEnabled} isInvalid={!value || mainModel?.base !== modelConfig?.base}>
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
};

export default memo(ParamControlAdapterModel);
