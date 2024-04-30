import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useControlAdapterIPMethod = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectControlAdaptersSlice, (controlAdapters) => {
        const ca = selectControlAdapterById(controlAdapters, id);
        assert(ca?.type === 'ip_adapter');
        return ca.method;
      }),
    [id]
  );

  const method = useAppSelector(selector);

  return method;
};
