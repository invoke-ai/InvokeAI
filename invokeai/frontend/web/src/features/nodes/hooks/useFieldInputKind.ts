import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useFieldInputKind = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, selectNodeTemplatesSlice, (nodes, nodeTemplates) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return;
        }
        const nodeTemplate = nodeTemplates.templates[node?.data.type ?? ''];
        const fieldTemplate = nodeTemplate?.inputs[fieldName];
        return fieldTemplate?.input;
      }),
    [fieldName, nodeId]
  );

  const inputKind = useAppSelector(selector);

  return inputKind;
};
