import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';

export const useNodeTemplate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes, nodeTemplates }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        const nodeTemplate = nodeTemplates.templates[node?.data.type ?? ''];
        return nodeTemplate;
      }),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
