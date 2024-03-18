import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';

export const useControlAdapterIsEnabled = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        selectControlAdaptersSlice,
        (controlAdapters) => selectControlAdapterById(controlAdapters, id)?.isEnabled ?? false
      ),
    [id]
  );

  const isEnabled = useAppSelector(selector);

  return isEnabled;
};
