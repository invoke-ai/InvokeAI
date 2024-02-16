import { PropsWithChildren, memo, useEffect } from 'react';
import { modelChanged } from '../src/features/parameters/store/generationSlice';
import { useAppDispatch } from '../src/app/store/storeHooks';
import { useGlobalModifiersInit } from '@invoke-ai/ui-library';
/**
 * Initializes some state for storybook. Must be in a different component
 * so that it is run inside the redux context.
 */
export const ReduxInit = memo((props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  useGlobalModifiersInit();
  useEffect(() => {
    dispatch(modelChanged({ key: 'test_model', base: 'sd-1' }));
  }, []);

  return props.children;
});

ReduxInit.displayName = 'ReduxInit';
