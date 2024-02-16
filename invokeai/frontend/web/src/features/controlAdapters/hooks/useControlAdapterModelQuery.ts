import type { ControlAdapterType } from 'features/controlAdapters/store/types';
import {
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetT2IAdapterModelsQuery,
} from 'services/api/endpoints/models';

export const useControlAdapterModelQuery = (type: ControlAdapterType) => {
  const controlNetModelsQuery = useGetControlNetModelsQuery();
  const t2iAdapterModelsQuery = useGetT2IAdapterModelsQuery();
  const ipAdapterModelsQuery = useGetIPAdapterModelsQuery();

  if (type === 'controlnet') {
    return controlNetModelsQuery;
  }
  if (type === 't2i_adapter') {
    return t2iAdapterModelsQuery;
  }
  if (type === 'ip_adapter') {
    return ipAdapterModelsQuery;
  }

  // Assert that the end of the function is not reachable.
  const exhaustiveCheck: never = type;
  return exhaustiveCheck;
};
