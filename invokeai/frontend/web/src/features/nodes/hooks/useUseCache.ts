import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useUseCache = () => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector(ctx.selectNodeDataSafe, (data) => {
        return data?.useCache ?? false;
      }),
    [ctx]
  );
  return useAppSelector(selector);
};
