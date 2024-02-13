import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import { KIND_MAP } from 'features/nodes/types/constants';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useFieldTemplateTitle = (nodeId: string, fieldName: string, kind: 'input' | 'output') => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, selectNodeTemplatesSlice, (nodes, nodeTemplates) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return;
        }
        const nodeTemplate = nodeTemplates.templates[node?.data.type ?? ''];
        return nodeTemplate?.[KIND_MAP[kind]][fieldName]?.title;
      }),
    [fieldName, kind, nodeId]
  );

  const fieldTemplateTitle = useAppSelector(selector);

  return fieldTemplateTitle;
};
