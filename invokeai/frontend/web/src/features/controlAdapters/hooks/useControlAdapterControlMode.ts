import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useMemo } from 'react';
import { selectControlAdapterById } from '../store/controlAdaptersSlice';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { isControlNet } from '../store/types';

export const useControlAdapterControlMode = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlAdapters }) => {
          const ca = selectControlAdapterById(controlAdapters, id);
          if (ca && isControlNet(ca)) {
            return ca.controlMode;
          }
          return undefined;
        },
        defaultSelectorOptions
      ),
    [id]
  );

  const controlMode = useAppSelector(selector);

  return controlMode;
};
