import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';

import { SelectItem } from '@mantine/core';
import { forEach } from 'lodash-es';
import { useListModelsQuery } from 'services/api/endpoints/models';

import { RootState } from 'app/store/store';
import { vaeSelected } from 'features/parameters/store/generationSlice';

export const MODEL_TYPE_MAP = {
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
};

const VAESelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: vaeModels } = useListModelsQuery({
    model_type: 'vae',
  });

  const selectedModelId = useAppSelector(
    (state: RootState) => state.generation.vae
  );

  const data = useMemo(() => {
    if (!vaeModels) {
      return [];
    }

    const data: SelectItem[] = [
      {
        value: 'none',
        label: 'None',
        group: 'Default',
      },
    ];

    forEach(vaeModels.entities, (model, id) => {
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
  }, [vaeModels]);

  const selectedModel = useMemo(
    () => vaeModels?.entities[selectedModelId],
    [vaeModels?.entities, selectedModelId]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(vaeSelected(v));
    },
    [dispatch]
  );

  useEffect(() => {
    if (selectedModelId && vaeModels?.ids.includes(selectedModelId)) {
      return;
    }
    handleChangeModel('none');
  }, [handleChangeModel, vaeModels?.ids, selectedModelId]);

  return (
    <IAIMantineSelect
      tooltip={selectedModel?.description}
      label={t('modelManager.customVAE')}
      value={selectedModelId}
      placeholder="Pick one"
      data={data}
      onChange={handleChangeModel}
    />
  );
};

export default memo(VAESelect);
