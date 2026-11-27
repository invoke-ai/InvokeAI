import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useOutputFieldTemplateExists = (fieldName: string) => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector(ctx.buildSelectOutputFieldTemplateSafe(fieldName), (fieldTemplate) => {
        return !!fieldTemplate;
      }),
    [ctx, fieldName]
  );
  return useAppSelector(selector);
};
