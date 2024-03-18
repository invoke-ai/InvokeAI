import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';
import { useGetModelConfigWithTypeGuard } from 'services/api/hooks/useGetModelConfigWithTypeGuard';
import { isControlAdapterModelConfig } from 'services/api/types';

export const useControlAdapterModel = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(
        selectControlAdaptersSlice,
        (controlAdapters) => selectControlAdapterById(controlAdapters, id)?.model?.key
      ),
    [id]
  );

  const key = useAppSelector(selector);

  const result = useGetModelConfigWithTypeGuard(key ?? skipToken, isControlAdapterModelConfig);

  return result;
};
