import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectInvocationNodeSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useOutputFieldTemplateExists = (nodeId: string, fieldName: string) => {
  const templates = useStore($templates);

  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = selectInvocationNodeSafe(nodesSlice, nodeId);
        if (!node) {
          return false;
        }
        const nodeTemplate = templates[node.data.type];
        const fieldTemplate = nodeTemplate?.outputs[fieldName];
        return Boolean(fieldTemplate);
      }),
    [fieldName, nodeId, templates]
  );

  const exists = useAppSelector(selector);

  return exists;
};
