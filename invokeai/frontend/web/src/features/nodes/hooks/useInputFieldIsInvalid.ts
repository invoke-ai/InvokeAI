import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectFieldInputInstance, selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { getFieldErrors } from 'features/nodes/store/util/fieldValidators';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useInputFieldIsInvalid = (nodeId: string, fieldName: string) => {
  const templates = useStore($templates);

  const selectIsInvalid = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = selectInvocationNode(nodes, nodeId);
        const field = selectFieldInputInstance(nodes, nodeId, fieldName);

        // No field instance is a problem - should not happen
        if (!field) {
          return true;
        }

        const nodeTemplate = templates[node.data.type];
        assert(nodeTemplate, `Template for input node type ${node.data.type} not found.`);

        const fieldTemplate = nodeTemplate.inputs[fieldName];
        assert(fieldTemplate, `Template for input field ${fieldName} not found.`);

        const errors = getFieldErrors(node, nodeTemplate, field, fieldTemplate, nodes);

        return errors.length > 0;
      }),
    [nodeId, fieldName, templates]
  );

  const isInvalid = useAppSelector(selectIsInvalid);

  return isInvalid;
};
