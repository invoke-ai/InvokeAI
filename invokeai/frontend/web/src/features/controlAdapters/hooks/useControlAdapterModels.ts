import type { ControlAdapterType } from 'features/controlAdapters/store/types';
import { useMemo } from 'react';
import {
  controlNetModelsAdapterSelectors,
  ipAdapterModelsAdapterSelectors,
  t2iAdapterModelsAdapterSelectors,
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetT2IAdapterModelsQuery,
} from 'services/api/endpoints/models';

export const useControlAdapterModels = (type?: ControlAdapterType) => {
  const { data: controlNetModelsData } = useGetControlNetModelsQuery();
  const controlNetModels = useMemo(
    () => (controlNetModelsData ? controlNetModelsAdapterSelectors.selectAll(controlNetModelsData) : []),
    [controlNetModelsData]
  );

  const { data: t2iAdapterModelsData } = useGetT2IAdapterModelsQuery();
  const t2iAdapterModels = useMemo(
    () => (t2iAdapterModelsData ? t2iAdapterModelsAdapterSelectors.selectAll(t2iAdapterModelsData) : []),
    [t2iAdapterModelsData]
  );
  const { data: ipAdapterModelsData } = useGetIPAdapterModelsQuery();
  const ipAdapterModels = useMemo(
    () => (ipAdapterModelsData ? ipAdapterModelsAdapterSelectors.selectAll(ipAdapterModelsData) : []),
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
  return [];
};
