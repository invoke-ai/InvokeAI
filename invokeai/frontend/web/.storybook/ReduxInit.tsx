import { PropsWithChildren, useEffect } from 'react';
import { modelChanged } from '../src/features/parameters/store/generationSlice';
import { useAppDispatch } from '../src/app/store/storeHooks';
import { useGlobalModifiersInit } from '../src/common/hooks/useGlobalModifiers';
/**
 * Initializes some state for storybook. Must be in a different component
 * so that it is run inside the redux context.
 */
export const ReduxInit = (props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  useGlobalModifiersInit();
  useEffect(() => {
    dispatch(
      modelChanged({
        model_name: 'test_model',
        base_model: 'sd-1',
        model_type: 'main',
      })
    );
  }, []);

  return props.children;
};
