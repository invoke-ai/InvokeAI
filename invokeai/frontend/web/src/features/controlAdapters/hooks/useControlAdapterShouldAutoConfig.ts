import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { selectControlAdapterById } from '../store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from '../store/types';

export const useControlAdapterShouldAutoConfig = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlAdapters }) => {
          const ca = selectControlAdapterById(controlAdapters, id);
          if (ca && isControlNetOrT2IAdapter(ca)) {
            return ca.shouldAutoConfig;
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
