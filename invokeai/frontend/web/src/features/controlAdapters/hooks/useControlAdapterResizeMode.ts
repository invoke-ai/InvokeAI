import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { useMemo } from 'react';

export const useControlAdapterResizeMode = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectControlAdaptersSlice, (controlAdapters) => {
        const ca = selectControlAdapterById(controlAdapters, id);
        if (ca && isControlNetOrT2IAdapter(ca)) {
          return ca.resizeMode;
        }
        return undefined;
      }),
    [id]
  );

  const controlMode = useAppSelector(selector);

  return controlMode;
};
