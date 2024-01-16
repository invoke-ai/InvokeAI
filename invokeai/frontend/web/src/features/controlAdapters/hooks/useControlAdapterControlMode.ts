import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNet } from 'features/controlAdapters/store/types';
import { useMemo } from 'react';

export const useControlAdapterControlMode = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectControlAdaptersSlice, (controlAdapters) => {
        const ca = selectControlAdapterById(controlAdapters, id);
        if (ca && isControlNet(ca)) {
          return ca.controlMode;
        }
        return undefined;
      }),
    [id]
  );

  const controlMode = useAppSelector(selector);

  return controlMode;
};
