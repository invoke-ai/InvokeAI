import { useGlobalModifiersInit } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo, useEffect } from 'react';

import { useAppDispatch } from '../src/app/store/storeHooks';
import { modelChanged } from '../src/features/controlLayers/store/paramsSlice';
/**
 * Initializes some state for storybook. Must be in a different component
 * so that it is run inside the redux context.
 */
export const ReduxInit = memo(({ children }: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  useGlobalModifiersInit();
  useEffect(() => {
    dispatch(
      modelChanged({ model: { key: 'test_model', hash: 'some_hash', name: 'some name', base: 'sd-1', type: 'main' } })
    );
  }, [dispatch]);

  return children;
});

ReduxInit.displayName = 'ReduxInit';
