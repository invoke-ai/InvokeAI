import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

/**
 * Gets the description of an input field for a given node.
 *
 * If the node doesn't exist or is not an invocation node, an empty string is returned.
 *
 * @param nodeId The ID of the node
 * @param fieldName The name of the field
 * @returns
 */
export const useInputFieldDescription = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return '';
        }
        return node?.data.inputs[fieldName]?.description ?? '';
      }),
    [fieldName, nodeId]
  );

  const description = useAppSelector(selector);
  return description;
};
