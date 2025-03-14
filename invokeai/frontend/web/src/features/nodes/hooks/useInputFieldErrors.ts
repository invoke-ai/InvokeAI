import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useDebouncedAppSelector } from 'app/store/use-debounced-app-selector';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectFieldInputInstance, selectInvocationNodeSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import { getFieldErrors } from 'features/nodes/store/util/fieldValidators';
import { useMemo } from 'react';
import { assert } from 'tsafe';

/**
 * A hook that returns the errors for a given input field. The errors calculation is debounced.
 *
 * @param nodeId The id of the node
 * @param fieldName The name of the field
 * @returns An array of FieldError objects
 */
export const useInputFieldErrors = (nodeId: string, fieldName: string) => {
  const templates = useStore($templates);

  const selectFieldErrors = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = selectInvocationNodeSafe(nodes, nodeId);
        if (!node) {
          // If the node is not found, return an empty array - might happen during node deletion
          return [];
        }
        const field = selectFieldInputInstance(nodes, nodeId, fieldName);

        const nodeTemplate = templates[node.data.type];
        assert(nodeTemplate, `Template for input node type ${node.data.type} not found.`);

        const fieldTemplate = nodeTemplate.inputs[fieldName];
        assert(fieldTemplate, `Template for input field ${fieldName} not found.`);

        return getFieldErrors(node, nodeTemplate, field, fieldTemplate, nodes);
      }),
    [nodeId, fieldName, templates]
  );

  const fieldErrors = useDebouncedAppSelector(selectFieldErrors);

  return fieldErrors;
};
