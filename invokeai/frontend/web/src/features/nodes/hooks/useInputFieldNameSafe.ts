import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useInputFieldNameSafe = (fieldName: string) => {
  const ctx = useInvocationNodeContext();

  const selector = useMemo(
    () =>
      createSelector(
        [ctx.buildSelectInputFieldSafe(fieldName), ctx.buildSelectInputFieldTemplateSafe(fieldName)],
        (fieldInstance, fieldTemplate) => {
          const name = fieldInstance?.label || fieldTemplate?.title || fieldName;
          return name;
        }
      ),
    [fieldName, ctx]
  );

  const name = useAppSelector(selector);

  return name;
};
