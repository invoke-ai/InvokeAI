import { useAppDispatch } from 'app/store/storeHooks';
import { controlNetProcessorParamsChanged } from 'features/controlNet/store/controlNetSlice';
import { ControlNetProcessorNode } from 'features/controlNet/store/types';
import { useCallback } from 'react';

export const useProcessorNodeChanged = () => {
  const dispatch = useAppDispatch();
  const handleProcessorNodeChanged = useCallback(
    (controlNetId: string, changes: Partial<ControlNetProcessorNode>) => {
      dispatch(
        controlNetProcessorParamsChanged({
          controlNetId,
          changes,
        })
      );
    },
    [dispatch]
  );
  return handleProcessorNodeChanged;
};
