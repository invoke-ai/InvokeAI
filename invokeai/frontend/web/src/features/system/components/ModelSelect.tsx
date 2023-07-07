import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';

import { SelectItem } from '@mantine/core';
import { RootState } from 'app/store/store';
import { modelSelected } from 'features/parameters/store/actions';
import { forEach, isString } from 'lodash-es';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

export const MODEL_TYPE_MAP = {
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
};

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const currentModel = useAppSelector(
    (state: RootState) => state.generation.model
  );

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
        label: model.name,
        group: MODEL_TYPE_MAP[model.base_model],
      });
    });

    return data;
  }, [mainModels]);

  const selectedModel = useMemo(
    () => mainModels?.entities[currentModel?.id || ''],
    [mainModels?.entities, currentModel]
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
    if (isLoading) {
      // return early here to avoid resetting model selection before we've loaded the available models
      return;
    }

    if (selectedModel && mainModels?.ids.includes(selectedModel?.id)) {
      // the selected model is an available model, no need to change it
      return;
    }

    const firstModel = mainModels?.ids[0];

    if (!isString(firstModel)) {
      return;
    }

    handleChangeModel(firstModel);
  }, [handleChangeModel, isLoading, mainModels?.ids, selectedModel]);

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
      placeholder={data.length > 0 ? 'Select a model' : 'No models detected!'}
      data={data}
      error={data.length === 0}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ModelSelect);
