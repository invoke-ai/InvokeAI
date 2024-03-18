import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectInvocationNode, selectNodeTemplate } from 'features/nodes/store/selectors';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import { useMemo } from 'react';

export const useNodeNeedsUpdate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const node = selectInvocationNode(nodes, nodeId);
        const template = selectNodeTemplate(nodes, nodeId);
        if (!node || !template) {
          return false;
        }
        return getNeedsUpdate(node, template);
      }),
    [nodeId]
  );

  const needsUpdate = useAppSelector(selector);

  return needsUpdate;
};
