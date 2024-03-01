import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

const selectModelKey = createSelector(selectGenerationSlice, (generation) => generation.model?.key);

export const useSelectedModelConfig = () => {
  const key = useAppSelector(selectModelKey);
  const { currentData: modelConfig } = useGetModelConfigQuery(key ?? skipToken);

  return modelConfig;
};
