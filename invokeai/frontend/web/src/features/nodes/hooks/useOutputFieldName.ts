import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useOutputFieldName = (fieldName: string) => {
  const ctx = useInvocationNodeContext();

  const selector = useMemo(
    () =>
      createSelector([ctx.buildSelectOutputFieldTemplateSafe(fieldName)], (fieldTemplate) => {
        const name = fieldTemplate?.title || fieldName;
        return name;
      }),
    [fieldName, ctx]
  );

  return useAppSelector(selector);
};
