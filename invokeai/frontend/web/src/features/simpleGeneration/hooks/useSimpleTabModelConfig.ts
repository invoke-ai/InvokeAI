import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectModelKey } from 'features/simpleGeneration/store/slice';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

export const useSimpleTabModelConfig = () => {
  const key = useAppSelector(selectModelKey);
  const { data: modelConfig } = useGetModelConfigQuery(key ?? skipToken);

  return modelConfig;
};
