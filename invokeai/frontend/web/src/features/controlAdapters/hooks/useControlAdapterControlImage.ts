import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';

export const useControlAdapterControlImage = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        selectControlAdaptersSlice,
        (controlAdapters) => selectControlAdapterById(controlAdapters, id)?.controlImage
      ),
    [id]
  );

  const controlImageName = useAppSelector(selector);

  return controlImageName;
};
