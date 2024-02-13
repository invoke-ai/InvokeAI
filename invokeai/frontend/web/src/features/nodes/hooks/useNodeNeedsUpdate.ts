import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import { useMemo } from 'react';

export const useNodeNeedsUpdate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, selectNodeTemplatesSlice, (nodes, nodeTemplates) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        const template = nodeTemplates.templates[node?.data.type ?? ''];
        if (isInvocationNode(node) && template) {
          return getNeedsUpdate(node, template);
        }
        return false;
      }),
    [nodeId]
  );

  const needsUpdate = useAppSelector(selector);

  return needsUpdate;
};
