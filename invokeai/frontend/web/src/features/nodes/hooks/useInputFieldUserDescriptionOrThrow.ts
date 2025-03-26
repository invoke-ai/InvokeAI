import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectFieldInputInstance, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

/**
 * Gets the user-defined description of an input field for a given node.
 *
 * If the node doesn't exist or is not an invocation node, an error is thrown.
 *
 * @param nodeId The ID of the node
 * @param fieldName The name of the field
 */
export const useInputFieldUserDescriptionOrThrow = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () => createSelector(selectNodesSlice, (nodes) => selectFieldInputInstance(nodes, nodeId, fieldName).description),
    [fieldName, nodeId]
  );

  const description = useAppSelector(selector);
  return description;
};
