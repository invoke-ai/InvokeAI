import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { selectControlAdapterById } from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';

export const useControlAdapterBeginEndStepPct = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ controlAdapters }) => {
        const cn = selectControlAdapterById(controlAdapters, id);
        return cn
          ? {
              beginStepPct: cn.beginStepPct,
              endStepPct: cn.endStepPct,
            }
          : undefined;
      }),
    [id]
  );

  const stepPcts = useAppSelector(selector);

  return stepPcts;
};
