import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';

export const useControlAdapterModel = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(
        selectControlAdaptersSlice,
        (controlAdapters) => selectControlAdapterById(controlAdapters, id)?.model
      ),
    [id]
  );

  const model = useAppSelector(selector);

  return model;
};
