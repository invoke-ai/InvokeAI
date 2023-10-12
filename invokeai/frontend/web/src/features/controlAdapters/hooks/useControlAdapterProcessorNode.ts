import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useMemo } from 'react';
import { selectControlAdapterById } from '../store/controlAdaptersSlice';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { isControlNetOrT2IAdapter } from '../store/types';

export const useControlAdapterProcessorNode = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlAdapters }) => {
          const ca = selectControlAdapterById(controlAdapters, id);

          return ca && isControlNetOrT2IAdapter(ca)
            ? ca.processorNode
            : undefined;
        },
        defaultSelectorOptions
      ),
    [id]
  );

  const processorNode = useAppSelector(selector);

  return processorNode;
};
