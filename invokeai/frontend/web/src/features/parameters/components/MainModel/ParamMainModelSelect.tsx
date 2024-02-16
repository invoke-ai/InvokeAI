import { Combobox, FormControl, FormLabel, Tooltip } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { modelSelected } from 'features/parameters/store/actions';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { pick } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { NON_REFINER_BASE_MODELS } from 'services/api/constants';
import type { MainModelConfig } from 'services/api/endpoints/models';
import { getModelId, mainModelsAdapterSelectors, useGetMainModelsQuery } from 'services/api/endpoints/models';

const selectModel = createMemoizedSelector(selectGenerationSlice, (generation) => generation.model);

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const model = useAppSelector(selectModel);
  const { data, isLoading } = useGetMainModelsQuery(NON_REFINER_BASE_MODELS);
  const tooltipLabel = useMemo(() => {
    if (!data || !model) {
      return;
    }
    return mainModelsAdapterSelectors.selectById(data, getModelId(model))?.description;
  }, [data, model]);
  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (!model) {
        return;
      }
      dispatch(modelSelected(pick(model, ['base_model', 'model_name', 'model_type'])));
    },
    [dispatch]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelEntities: data,
    onChange: _onChange,
    selectedModel: model,
    isLoading,
  });

  return (
    <Tooltip label={tooltipLabel}>
      <FormControl isDisabled={!options.length} isInvalid={!options.length}>
        <FormLabel>{t('modelManager.model')}</FormLabel>
        <Combobox
          value={value}
          placeholder={placeholder}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    </Tooltip>
  );
};

export default memo(ParamMainModelSelect);
