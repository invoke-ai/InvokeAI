import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useInputFieldInstanceExists = (fieldName: string): boolean => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector(ctx.buildSelectInputFieldSafe(fieldName), (field) => {
        return !!field;
      }),
    [ctx, fieldName]
  );
  return useAppSelector(selector);
};
