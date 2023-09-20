import { SelectItem } from '@mantine/core';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { ipAdapterModelChanged } from 'features/controlNet/store/controlNetSlice';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToIPAdapterModelParam } from 'features/parameters/util/modelIdToIPAdapterModelParams';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetIPAdapterModelsQuery } from 'services/api/endpoints/models';

const ParamIPAdapterModelSelect = () => {
  const ipAdapterModel = useAppSelector(
    (state: RootState) => state.controlNet.ipAdapterInfo.model
  );
  const model = useAppSelector((state: RootState) => state.generation.model);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { data: ipAdapterModels } = useGetIPAdapterModelsQuery();

  // grab the full model entity from the RTK Query cache
  const selectedModel = useMemo(
    () =>
      ipAdapterModels?.entities[
        `${ipAdapterModel?.base_model}/ip_adapter/${ipAdapterModel?.model_name}`
      ] ?? null,
    [
      ipAdapterModel?.base_model,
      ipAdapterModel?.model_name,
      ipAdapterModels?.entities,
    ]
  );

  const data = useMemo(() => {
    if (!ipAdapterModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(ipAdapterModels.entities, (ipAdapterModel, id) => {
      if (!ipAdapterModel) {
        return;
      }

      const disabled = model?.base_model !== ipAdapterModel.base_model;

      data.push({
        value: id,
        label: ipAdapterModel.model_name,
        group: MODEL_TYPE_MAP[ipAdapterModel.base_model],
        disabled,
        tooltip: disabled
          ? `Incompatible base model: ${ipAdapterModel.base_model}`
          : undefined,
      });
    });

    return data.sort((a, b) => (a.disabled && !b.disabled ? 1 : -1));
  }, [ipAdapterModels, model?.base_model]);

  const handleValueChanged = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newIPAdapterModel = modelIdToIPAdapterModelParam(v);

      if (!newIPAdapterModel) {
        return;
      }

      dispatch(ipAdapterModelChanged(newIPAdapterModel));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      label={t('controlnet.ipAdapterModel')}
      className="nowheel nodrag"
      tooltip={selectedModel?.description}
      value={selectedModel?.id ?? null}
      placeholder="Pick one"
      error={!selectedModel}
      data={data}
      onChange={handleValueChanged}
      sx={{ width: '100%' }}
    />
  );
};

export default memo(ParamIPAdapterModelSelect);
