import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';

import { SelectItem } from '@mantine/core';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { modelIdToMainModelField } from 'features/nodes/util/modelIdToMainModelField';
import { modelSelected } from 'features/parameters/store/actions';
import { forEach } from 'lodash-es';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

export const MODEL_TYPE_MAP = {
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
};

const selector = createSelector(
  stateSelector,
  (state) => ({ currentModel: state.generation.model }),
  defaultSelectorOptions
);

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { currentModel } = useAppSelector(selector);

  const { data: mainModels, isLoading } = useGetMainModelsQuery();

  const data = useMemo(() => {
    if (!mainModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(mainModels.entities, (model, id) => {
      if (!model) {
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

  const selectedModel = useMemo(
    () =>
      mainModels?.entities[
        `${currentModel?.base_model}/main/${currentModel?.model_name}`
      ],
    [mainModels?.entities, currentModel]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const modelField = modelIdToMainModelField(v);
      dispatch(modelSelected(modelField));
    },
    [dispatch]
  );

  return isLoading ? (
    <IAIMantineSelect
      label={t('modelManager.model')}
      placeholder="Loading..."
      disabled={true}
      data={[]}
    />
  ) : (
    <IAIMantineSelect
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

export default memo(ModelSelect);
