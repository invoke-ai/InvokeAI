import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useMemo } from 'react';
import { selectControlAdapterById } from '../store/controlAdaptersSlice';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

export const useControlAdapterBeginEndStepPct = (id: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlAdapters }) => {
          const cn = selectControlAdapterById(controlAdapters, id);
          return cn
            ? {
                beginStepPct: cn.beginStepPct,
                endStepPct: cn.endStepPct,
              }
            : undefined;
        },
        defaultSelectorOptions
      ),
    [id]
  );

  const stepPcts = useAppSelector(selector);

  return stepPcts;
};
