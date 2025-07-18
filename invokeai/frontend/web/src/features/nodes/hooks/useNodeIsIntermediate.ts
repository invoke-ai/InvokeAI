import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useNodeIsIntermediate = (): boolean => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector(ctx.selectNodeDataSafe, (data) => {
        return data?.isIntermediate ?? false;
      }),
    [ctx]
  );
  return useAppSelector(selector);
};
