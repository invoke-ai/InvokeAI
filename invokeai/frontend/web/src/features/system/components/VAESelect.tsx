import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';

import { SelectItem } from '@mantine/core';
import { forEach } from 'lodash-es';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

import { RootState } from 'app/store/store';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { vaeSelected } from 'features/parameters/store/generationSlice';
import { zVaeModel } from 'features/parameters/store/parameterZodSchemas';
import { MODEL_TYPE_MAP } from './ModelSelect';

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
        value: 'default',
        label: 'Default',
        group: 'Default',
      },
    ];

    forEach(vaeModels.entities, (model, id) => {
      if (!model) {
        return;
      }

      const disabled = currentMainModel?.base_model !== model.base_model;

      data.push({
        value: id,
        label: model.name,
        group: MODEL_TYPE_MAP[model.base_model],
        disabled,
        tooltip: disabled
          ? `Incompatible base model: ${model.base_model}`
          : undefined,
      });
    });

    return data.sort((a, b) => (a.disabled && !b.disabled ? 1 : -1));
  }, [vaeModels, currentMainModel?.base_model]);

  const selectedVaeModel = useMemo(
    () => (selectedVae?.id ? vaeModels?.entities[selectedVae?.id] : null),
    [vaeModels?.entities, selectedVae]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v || v === 'default') {
        dispatch(vaeSelected(null));
        return;
      }

      const [base_model, type, name] = v.split('/');

      const model = zVaeModel.parse({
        id: v,
        name,
        base_model,
      });

      dispatch(vaeSelected(model));
    },
    [dispatch]
  );

  useEffect(() => {
    if (selectedVae && vaeModels?.ids.includes(selectedVae.id)) {
      return;
    }
    dispatch(vaeSelected(null));
  }, [handleChangeModel, vaeModels?.ids, selectedVae, dispatch]);

  return (
    <IAIMantineSelect
      itemComponent={IAIMantineSelectItemWithTooltip}
      tooltip={selectedVaeModel?.description}
      label={t('modelManager.vae')}
      value={selectedVae?.id ?? 'default'}
      placeholder="Default"
      data={data}
      onChange={handleChangeModel}
      disabled={data.length === 0}
      clearable
    />
  );
};

export default memo(VAESelect);
