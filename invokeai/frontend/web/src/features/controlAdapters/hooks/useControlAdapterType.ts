import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useControlAdapterType = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectControlAdaptersSlice, (controlAdapters) => {
        const type = selectControlAdapterById(controlAdapters, id)?.type;
        assert(type !== undefined, `Control adapter with id ${id} not found`);
        return type;
      }),
    [id]
  );

  const type = useAppSelector(selector);

  return type;
};
