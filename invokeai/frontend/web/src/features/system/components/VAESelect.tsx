import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';

import { SelectItem } from '@mantine/core';
import { forEach } from 'lodash-es';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

import { RootState } from 'app/store/store';
import { vaeSelected } from 'features/parameters/store/generationSlice';
import { MODEL_TYPE_MAP } from './ModelSelect';

const VAESelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: vaeModels } = useGetVaeModelsQuery();

  const selectedModelId = useAppSelector(
    (state: RootState) => state.generation.vae
  );

  const data = useMemo(() => {
    if (!vaeModels) {
      return [];
    }

    const data: SelectItem[] = [
      {
        value: 'auto',
        label: 'Automatic',
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
    handleChangeModel('auto');
  }, [handleChangeModel, vaeModels?.ids, selectedModelId]);

  return (
    <IAIMantineSelect
      tooltip={selectedModel?.description}
      label={t('modelManager.vae')}
      value={selectedModelId}
      placeholder="Pick one"
      data={data}
      onChange={handleChangeModel}
    />
  );
};

export default memo(VAESelect);
