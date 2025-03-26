import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectFieldInputInstanceSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

/**
 * Gets the user-defined title of an input field for a given node.
 *
 * If the node doesn't exist or is not an invocation node, an empty string is returned.
 *
 * @param nodeId The ID of the node
 * @param fieldName The name of the field
 */
export const useInputFieldUserTitleSafe = (nodeId: string, fieldName: string): string => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => selectFieldInputInstanceSafe(nodes, nodeId, fieldName)?.label ?? ''),
    [fieldName, nodeId]
  );

  const title = useAppSelector(selector);

  return title;
};
