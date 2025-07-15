import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

/**
 * Gets the user-defined title of an input field for a given node.
 *
 * If the node doesn't exist or is not an invocation node, an error is thrown.
 *
 * @param fieldName The name of the field
 */
export const useInputFieldUserTitleOrThrow = (fieldName: string): string => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () => createSelector(ctx.buildSelectInputFieldOrThrow(fieldName), (field) => field.label),
    [ctx, fieldName]
  );
  return useAppSelector(selector);
};
