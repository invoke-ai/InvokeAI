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
import IAIMantineSelectItemWithTooltip from '../../../common/components/IAIMantineSelectItemWithTooltip';

const VAESelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: vaeModels } = useGetVaeModelsQuery();

  const currentMainModel = useAppSelector(
    (state: RootState) => state.generation.model
  );

  const selectedVae = useAppSelector(
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
        ...(currentMainModel?.base_model !== model.base_model
          ? { disabled: true, tooltip: 'Incompatible base model' }
          : {}),
      });
    });

    return data;
  }, [vaeModels, currentMainModel?.base_model]);

  const selectedVaeModel = useMemo(
    () => vaeModels?.entities[selectedVae],
    [vaeModels?.entities, selectedVae]
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
    if (selectedVae && vaeModels?.ids.includes(selectedVae)) {
      return;
    }
    handleChangeModel('auto');
  }, [handleChangeModel, vaeModels?.ids, selectedVae]);

  return (
    <IAIMantineSelect
      itemComponent={IAIMantineSelectItemWithTooltip}
      tooltip={selectedVaeModel?.description}
      label={t('modelManager.vae')}
      value={selectedVae}
      placeholder="Pick one"
      data={data}
      onChange={handleChangeModel}
    />
  );
};

export default memo(VAESelect);
