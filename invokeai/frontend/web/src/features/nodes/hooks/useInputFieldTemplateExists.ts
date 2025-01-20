import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useInputFieldTemplateExists = (nodeId: string, fieldName: string) => {
  const templates = useStore($templates);

  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = selectInvocationNode(nodesSlice, nodeId);
        const nodeTemplate = templates[node.data.type];
        const fieldTemplate = nodeTemplate?.inputs[fieldName];
        return Boolean(fieldTemplate);
      }),
    [fieldName, nodeId, templates]
  );

  const exists = useAppSelector(selector);

  return exists;
};
