import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { modelSelected } from 'features/parameters/store/generationSlice';

import { forEach, isString } from 'lodash-es';
import { SelectItem } from '@mantine/core';
import { RootState } from 'app/store/store';
import { useListModelsQuery } from 'services/api/endpoints/models';

export const MODEL_TYPE_MAP = {
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
};

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const selectedModelId = useAppSelector(
    (state: RootState) => state.generation.model
  );

  const { data: pipelineModels, isLoading } = useListModelsQuery({
    model_type: 'main',
  });

  const data = useMemo(() => {
    if (!pipelineModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(pipelineModels.entities, (model, id) => {
      if (!model) {
        return;
      }

      data.push({
        value: id,
        label: model.name,
        group: MODEL_TYPE_MAP[model.base_model],
      });
    });

    return data;
  }, [pipelineModels]);

  const selectedModel = useMemo(
    () => pipelineModels?.entities[selectedModelId],
    [pipelineModels?.entities, selectedModelId]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(modelSelected(v));
    },
    [dispatch]
  );

  useEffect(() => {
    // If the selected model is not in the list of models, select the first one
    // Handles first-run setting of models, and the user deleting the previously-selected model
    if (selectedModelId && pipelineModels?.ids.includes(selectedModelId)) {
      return;
    }

    const firstModel = pipelineModels?.ids[0];

    if (!isString(firstModel)) {
      return;
    }

    handleChangeModel(firstModel);
  }, [handleChangeModel, pipelineModels?.ids, selectedModelId]);

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
      value={selectedModelId}
      placeholder={data.length > 0 ? 'Select a model' : 'No models detected!'}
      data={data}
      error={data.length === 0}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ModelSelect);
