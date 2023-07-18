import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';

import { SelectItem } from '@mantine/core';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { modelSelected } from 'features/parameters/store/actions';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToMainModelParam } from 'features/parameters/util/modelIdToMainModelParam';
import { forEach } from 'lodash-es';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

const selector = createSelector(
  stateSelector,
  (state) => ({ model: state.generation.model }),
  defaultSelectorOptions
);

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { model } = useAppSelector(selector);

  const { data: mainModels, isLoading } = useGetMainModelsQuery();

  const data = useMemo(() => {
    if (!mainModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(mainModels.entities, (model, id) => {
      if (!model || ['sdxl', 'sdxl-refiner'].includes(model.base_model)) {
        return;
      }

      data.push({
        value: id,
        label: model.model_name,
        group: MODEL_TYPE_MAP[model.base_model],
      });
    });

    return data;
  }, [mainModels]);

  // grab the full model entity from the RTK Query cache
  // TODO: maybe we should just store the full model entity in state?
  const selectedModel = useMemo(
    () =>
      mainModels?.entities[`${model?.base_model}/main/${model?.model_name}`] ??
      null,
    [mainModels?.entities, model]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newModel = modelIdToMainModelParam(v);

      if (!newModel) {
        return;
      }

      dispatch(modelSelected(newModel));
    },
    [dispatch]
  );

  return isLoading ? (
    <IAIMantineSearchableSelect
      label={t('modelManager.model')}
      placeholder="Loading..."
      disabled={true}
      data={[]}
    />
  ) : (
    <IAIMantineSearchableSelect
      tooltip={selectedModel?.description}
      label={t('modelManager.model')}
      value={selectedModel?.id}
      placeholder={data.length > 0 ? 'Select a model' : 'No models available'}
      data={data}
      error={data.length === 0}
      disabled={data.length === 0}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ParamMainModelSelect);
