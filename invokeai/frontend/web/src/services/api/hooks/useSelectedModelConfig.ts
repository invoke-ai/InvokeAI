import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectModelKey } from 'features/controlLayers/store/paramsSlice';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

export const useSelectedModelConfig = () => {
  const key = useAppSelector(selectModelKey);
  const { currentData: modelConfig } = useGetModelConfigQuery(key ?? skipToken);

  return modelConfig;
};
