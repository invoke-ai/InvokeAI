import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { modelSelected } from 'features/parameters/store/generationSlice';

import { forEach, isString } from 'lodash-es';
import { SelectItem } from '@mantine/core';
import { RootState } from 'app/store/store';
import { useListModelsQuery } from 'services/apiSlice';

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

  const { data: pipelineModels } = useListModelsQuery({
    model_type: 'pipeline',
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
    if (selectedModelId && pipelineModels?.ids.includes(selectedModelId)) {
      return;
    }

    const firstModel = pipelineModels?.ids[0];

    if (!isString(firstModel)) {
      return;
    }

    handleChangeModel(firstModel);
  }, [handleChangeModel, pipelineModels?.ids, selectedModelId]);

  return (
    <IAIMantineSelect
      tooltip={selectedModel?.description}
      label={t('modelManager.model')}
      value={selectedModelId}
      placeholder="Pick one"
      data={data}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ModelSelect);
