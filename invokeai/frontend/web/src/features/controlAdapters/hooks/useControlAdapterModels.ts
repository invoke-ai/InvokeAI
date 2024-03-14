import type { ControlAdapterType } from 'features/controlAdapters/store/types';
import { useControlNetModels, useIPAdapterModels, useT2IAdapterModels } from 'services/api/hooks/modelsByType';

export const useControlAdapterModels = (type: ControlAdapterType) => {
  const controlNetModels = useControlNetModels();
  const t2iAdapterModels = useT2IAdapterModels();
  const ipAdapterModels = useIPAdapterModels();

  if (type === 'controlnet') {
    return controlNetModels;
  }
  if (type === 't2i_adapter') {
    return t2iAdapterModels;
  }
  if (type === 'ip_adapter') {
    return ipAdapterModels;
  }

  // Assert that the end of the function is not reachable.
  const exhaustiveCheck: never = type;
  return exhaustiveCheck;
};
