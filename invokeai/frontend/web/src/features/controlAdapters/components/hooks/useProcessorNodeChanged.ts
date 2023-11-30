import { useAppDispatch } from 'app/store/storeHooks';
import { controlAdapterProcessorParamsChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { ControlAdapterProcessorNode } from 'features/controlAdapters/store/types';
import { useCallback } from 'react';

export const useProcessorNodeChanged = () => {
  const dispatch = useAppDispatch();
  const handleProcessorNodeChanged = useCallback(
    (id: string, params: Partial<ControlAdapterProcessorNode>) => {
      dispatch(
        controlAdapterProcessorParamsChanged({
          id,
          params,
        })
      );
    },
    [dispatch]
  );
  return handleProcessorNodeChanged;
};
