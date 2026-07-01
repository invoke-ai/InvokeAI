import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useNodeTemplateTitleSafe = (): string | null => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () => createSelector(ctx.selectNodeTemplateSafe, (template) => template?.title ?? ''),
    [ctx]
  );
  return useAppSelector(selector);
};
