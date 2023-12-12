import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { selectControlAdapterById } from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNet } from 'features/controlAdapters/store/types';
import { useMemo } from 'react';

export const useControlAdapterControlMode = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ controlAdapters }) => {
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
