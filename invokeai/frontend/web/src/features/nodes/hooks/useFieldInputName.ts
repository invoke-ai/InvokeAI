import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useFieldInputName = (nodeId: string, fieldName: string) => {
  const templates = useStore($templates);

  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = selectInvocationNode(nodesSlice, nodeId);
        const instance = node.data.inputs[fieldName];
        const nodeTemplate = templates[node.data.type];
        const fieldTemplate = nodeTemplate?.inputs[fieldName];
        const name = instance?.label || fieldTemplate?.title || fieldName;
        return name;
      }),
    [fieldName, nodeId, templates]
  );

  const name = useAppSelector(selector);

  return name;
};
