import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { selectControlAdapterById } from 'features/controlAdapters/store/controlAdaptersSlice';
import { useMemo } from 'react';

export const useControlAdapterIsEnabled = (id: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(
        stateSelector,
        ({ controlAdapters }) =>
          selectControlAdapterById(controlAdapters, id)?.isEnabled ?? false
      ),
    [id]
  );

  const isEnabled = useAppSelector(selector);

  return isEnabled;
};
