import type { ControlAdapterType } from 'features/controlAdapters/store/types';
import {
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetT2IAdapterModelsQuery,
} from 'services/api/endpoints/models';

export const useControlAdapterModelEntities = (type?: ControlAdapterType) => {
  const { data: controlNetModelsData } = useGetControlNetModelsQuery();
  const { data: t2iAdapterModelsData } = useGetT2IAdapterModelsQuery();
  const { data: ipAdapterModelsData } = useGetIPAdapterModelsQuery();

  if (type === 'controlnet') {
    return controlNetModelsData;
  }
  if (type === 't2i_adapter') {
    return t2iAdapterModelsData;
  }
  if (type === 'ip_adapter') {
    return ipAdapterModelsData;
  }
  return;
};
