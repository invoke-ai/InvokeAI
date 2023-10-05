import { useAppDispatch } from 'app/store/storeHooks';
import { controlAdapterProcessorParamsChanged } from 'features/controlNet/store/controlAdaptersSlice';
import { ControlAdapterProcessorNode } from 'features/controlNet/store/types';
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
