import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

export const useSelectedModelConfig = () => {
  const key = useAppSelector((s) => s.canvasV2.params.model?.key);
  const { currentData: modelConfig } = useGetModelConfigQuery(key ?? skipToken);

  return modelConfig;
};
