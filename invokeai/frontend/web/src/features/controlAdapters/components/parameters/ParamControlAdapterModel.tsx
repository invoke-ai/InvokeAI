import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterModel } from 'features/controlAdapters/hooks/useControlAdapterModel';
import { useControlAdapterModelEntities } from 'features/controlAdapters/hooks/useControlAdapterModelEntities';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import { controlAdapterModelChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { pick } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig, ControlNetConfig, IPAdapterConfig, T2IAdapterConfig } from 'services/api/types';

type ParamControlAdapterModelProps = {
  id: string;
};

const selectMainModel = createMemoizedSelector(selectGenerationSlice, (generation) => generation.model);

const ParamControlAdapterModel = ({ id }: ParamControlAdapterModelProps) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const controlAdapterType = useControlAdapterType(id);
  const model = useControlAdapterModel(id);
  const dispatch = useAppDispatch();
  const currentBaseModel = useAppSelector((s) => s.generation.model?.base);
  const mainModel = useAppSelector(selectMainModel);
  const { t } = useTranslation();

  const models = useControlAdapterModelEntities(controlAdapterType);

  const _onChange = useCallback(
    (model: ControlNetConfig | IPAdapterConfig | T2IAdapterConfig | null) => {
      if (!model) {
        return;
      }
      dispatch(
        controlAdapterModelChanged({
          id,
          model: pick(model, 'base', 'key'),
        })
      );
    },
    [dispatch, id]
  );

  const selectedModel = useMemo(
    () => (model && controlAdapterType ? { ...model, model_type: controlAdapterType } : null),
    [controlAdapterType, model]
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
    modelEntities: models,
    onChange: _onChange,
    selectedModel,
    getIsDisabled,
  });

  return (
    <Tooltip label={value?.description}>
      <FormControl isDisabled={!isEnabled} isInvalid={!value || mainModel?.base !== model?.base}>
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
