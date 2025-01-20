import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useOutputFieldName = (nodeId: string, fieldName: string) => {
  const templates = useStore($templates);

  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = selectInvocationNode(nodesSlice, nodeId);
        const nodeTemplate = templates[node.data.type];
        const fieldTemplate = nodeTemplate?.outputs[fieldName];
        const name = fieldTemplate?.title || fieldName;
        return name;
      }),
    [fieldName, nodeId, templates]
  );

  const name = useAppSelector(selector);

  return name;
};
