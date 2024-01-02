import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { selectControlAdapterById } from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';

export const useControlAdapter = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ controlAdapters }) =>
        selectControlAdapterById(controlAdapters, id)
      ),
    [id]
  );

  const controlAdapter = useAppSelector(selector);

  return controlAdapter;
};
