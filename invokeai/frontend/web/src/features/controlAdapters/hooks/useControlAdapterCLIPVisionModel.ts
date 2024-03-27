import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterById,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';

export const useControlAdapterCLIPVisionModel = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectControlAdaptersSlice, (controlAdapters) => {
        const cn = selectControlAdapterById(controlAdapters, id);
        if (cn && cn?.type === 'ip_adapter') {
          return cn.clipVisionModel;
        }
      }),
    [id]
  );

  const clipVisionModel = useAppSelector(selector);

  return clipVisionModel;
};
