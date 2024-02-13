import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import { useMemo } from 'react';

export const useNodeTemplate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, selectNodeTemplatesSlice, (nodes, nodeTemplates) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        const nodeTemplate = nodeTemplates.templates[node?.data.type ?? ''];
        return nodeTemplate;
      }),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
