import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

/**
 * Gets the user-defined description of an input field for a given node.
 *
 * If the node doesn't exist or is not an invocation node, an empty string is returned.
 *
 * @param fieldName The name of the field
 */
export const useInputFieldUserDescriptionSafe = (fieldName: string) => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () => createSelector(ctx.buildSelectInputFieldSafe(fieldName), (field) => field?.description ?? ''),
    [ctx, fieldName]
  );
  return useAppSelector(selector);
};
