import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useInputFieldTemplateTitleOrThrow = (fieldName: string): string => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () => createSelector(ctx.buildSelectInputFieldTemplateOrThrow(fieldName), (fieldTemplate) => fieldTemplate.title),
    [ctx, fieldName]
  );
  return useAppSelector(selector);
};
