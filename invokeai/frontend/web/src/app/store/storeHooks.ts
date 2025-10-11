import type { AppStore, AppThunkDispatch, RootState } from 'app/store/store';
import { useIsCanvasWorkflow } from 'app/store/workflowContext';
import { injectCanvasWorkflowKey, injectNodesWorkflowKey } from 'features/nodes/store/actionRouter';
import { useCallback } from 'react';
import type { TypedUseSelectorHook } from 'react-redux';
import { useDispatch, useSelector, useStore } from 'react-redux';

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch = (): AppThunkDispatch => {
  const isCanvasWorkflow = useIsCanvasWorkflow();
  const dispatch = useDispatch<AppThunkDispatch>();

  return useCallback(
    ((action: Parameters<AppThunkDispatch>[0]) => {
      // Inject workflow routing metadata into actions
      if (typeof action === 'object' && action !== null && 'type' in action) {
        if (isCanvasWorkflow) {
          injectCanvasWorkflowKey(action);
        } else {
          injectNodesWorkflowKey(action);
        }
      }

      return dispatch(action);
    }) as AppThunkDispatch,
    [dispatch, isCanvasWorkflow]
  );
};

export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
export const useAppStore = () => useStore.withTypes<AppStore>()();
