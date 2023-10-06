import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useMemo } from 'react';
import { selectControlAdapterById } from '../store/controlAdaptersSlice';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

export const useControlAdapterIsEnabled = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlAdapters }) =>
          selectControlAdapterById(controlAdapters, id)?.isEnabled ?? false,
        defaultSelectorOptions
      ),
    [id]
  );

  const isEnabled = useAppSelector(selector);

  return isEnabled;
};
