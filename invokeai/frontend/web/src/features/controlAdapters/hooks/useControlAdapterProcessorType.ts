import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useMemo } from 'react';
import { selectControlAdapterById } from 'features/controlAdapters/store/controlAdaptersSlice';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';

export const useControlAdapterProcessorType = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlAdapters }) => {
          const ca = selectControlAdapterById(controlAdapters, id);

          return ca && isControlNetOrT2IAdapter(ca)
            ? ca.processorType
            : undefined;
        },
        defaultSelectorOptions
      ),
    [id]
  );

  const processorType = useAppSelector(selector);

  return processorType;
};
