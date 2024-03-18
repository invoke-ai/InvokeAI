import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeData } from 'features/nodes/store/selectors';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeData = (nodeId: string): InvocationNodeData | null => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        return selectNodeData(nodes, nodeId);
      }),
    [nodeId]
  );

  const nodeData = useAppSelector(selector);

  return nodeData;
};
