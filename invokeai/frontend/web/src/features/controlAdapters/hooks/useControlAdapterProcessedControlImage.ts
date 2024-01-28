import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { useMemo } from 'react';

export const useControlAdapterProcessedControlImage = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectControlAdaptersSlice, (controlAdapters) => {
        const ca = selectControlAdapterById(controlAdapters, id);

        return ca && isControlNetOrT2IAdapter(ca) ? ca.processedControlImage : undefined;
      }),
    [id]
  );

  const weight = useAppSelector(selector);

  return weight;
};
