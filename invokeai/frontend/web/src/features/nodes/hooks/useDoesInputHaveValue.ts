import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useDoesInputHaveValue = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return;
        }
        return node?.data.inputs[fieldName]?.value !== undefined;
      }),
    [fieldName, nodeId]
  );

  const doesFieldHaveValue = useAppSelector(selector);

  return doesFieldHaveValue;
};
