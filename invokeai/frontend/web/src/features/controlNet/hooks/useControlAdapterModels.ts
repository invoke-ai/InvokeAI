import { useMemo } from 'react';
import {
  controlNetModelsAdapter,
  ipAdapterModelsAdapter,
  t2iAdapterModelsAdapter,
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetT2IAdapterModelsQuery,
} from 'services/api/endpoints/models';
import { ControlAdapterType } from '../store/types';

export const useControlAdapterModels = (type?: ControlAdapterType) => {
  const { data: controlNetModelsData } = useGetControlNetModelsQuery();
  const controlNetModels = useMemo(
    () =>
      controlNetModelsData
        ? controlNetModelsAdapter.getSelectors().selectAll(controlNetModelsData)
        : [],
    [controlNetModelsData]
  );
  const { data: t2iAdapterModelsData } = useGetT2IAdapterModelsQuery();
  const t2iAdapterModels = useMemo(
    () =>
      t2iAdapterModelsData
        ? t2iAdapterModelsAdapter.getSelectors().selectAll(t2iAdapterModelsData)
        : [],
    [t2iAdapterModelsData]
  );
  const { data: ipAdapterModelsData } = useGetIPAdapterModelsQuery();
  const ipAdapterModels = useMemo(
    () =>
      ipAdapterModelsData
        ? ipAdapterModelsAdapter.getSelectors().selectAll(ipAdapterModelsData)
        : [],
    [ipAdapterModelsData]
  );

  if (type === 'controlnet') {
    return controlNetModels;
  }
  if (type === 't2i_adapter') {
    return t2iAdapterModels;
  }
  if (type === 'ip_adapter') {
    return ipAdapterModels;
  }
  return;
};
