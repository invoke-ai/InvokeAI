import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';

import { SelectItem } from '@mantine/core';
import { forEach } from 'lodash-es';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { vaeSelected } from 'features/parameters/store/generationSlice';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToVAEModelParam } from 'features/parameters/util/modelIdToVAEModelParam';

const selector = createSelector(
  stateSelector,
  ({ generation }) => {
    const { model, vae } = generation;
    return { model, vae };
  },
  defaultSelectorOptions
);

const ParamVAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { model, vae } = useAppSelector(selector);

  const { data: vaeModels } = useGetVaeModelsQuery();

  const data = useMemo(() => {
    if (!vaeModels) {
      return [];
    }

    // add a "default" option, this means use the main model's included VAE
    const data: SelectItem[] = [
      {
        value: 'default',
        label: 'Default',
        group: 'Default',
      },
    ];

    forEach(vaeModels.entities, (vae, id) => {
      if (!vae) {
        return;
      }

      const disabled = model?.base_model !== vae.base_model;

      data.push({
        value: id,
        label: vae.model_name,
        group: MODEL_TYPE_MAP[vae.base_model],
        disabled,
        tooltip: disabled
          ? `Incompatible base model: ${vae.base_model}`
          : undefined,
      });
    });

    return data.sort((a, b) => (a.disabled && !b.disabled ? 1 : -1));
  }, [vaeModels, model?.base_model]);

  // grab the full model entity from the RTK Query cache
  const selectedVaeModel = useMemo(
    () =>
      vaeModels?.entities[`${vae?.base_model}/vae/${vae?.model_name}`] ?? null,
    [vaeModels?.entities, vae]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v || v === 'default') {
        dispatch(vaeSelected(null));
        return;
      }

      const newVaeModel = modelIdToVAEModelParam(v);

      if (!newVaeModel) {
        return;
      }

      dispatch(vaeSelected(newVaeModel));
    },
    [dispatch]
  );

  return (
    <IAIMantineSearchableSelect
      itemComponent={IAIMantineSelectItemWithTooltip}
      tooltip={selectedVaeModel?.description}
      label={t('modelManager.vae')}
      value={selectedVaeModel?.id ?? 'default'}
      placeholder="Default"
      data={data}
      onChange={handleChangeModel}
      disabled={data.length === 0}
      clearable
    />
  );
};

export default memo(ParamVAEModelSelect);
