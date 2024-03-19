import { useAppSelector } from 'app/store/storeHooks';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { ControlAdapterProcessorType } from 'features/controlAdapters/store/types';
import { useMemo } from 'react';

export const useGetDefaultForControlnetProcessor = (processorType: ControlAdapterProcessorType) => {
  const baseModel = useAppSelector((s) => s.generation.model?.base);

  const defaults = useMemo(() => {
    return CONTROLNET_PROCESSORS[processorType].buildDefaults(baseModel);
  }, [baseModel, processorType]);

  return defaults;
};
