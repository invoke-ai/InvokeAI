import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';
import type { Node } from 'reactflow';

export const useNode = (nodeId: string): Node => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        return selectNode(nodes, nodeId);
      }),
    [nodeId]
  );

  const node = useAppSelector(selector);

  return node;
};
