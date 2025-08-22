import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectModelKey } from 'features/controlLayers/store/paramsSlice';
import { selectVideoModelKey } from 'features/parameters/store/videoSlice';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';
import type { VideoApiModelConfig } from 'services/api/types';

export const useSelectedModelConfig = () => {
  const key = useAppSelector(selectModelKey);
  const { data: modelConfig } = useGetModelConfigQuery(key ?? skipToken);

  return modelConfig;
};

export const useSelectedVideoModelConfig = () => {
  const key = useAppSelector(selectVideoModelKey);
  const { data: modelConfig } = useGetModelConfigQuery(key ?? skipToken);

  return modelConfig as VideoApiModelConfig | undefined;
};
