import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useNodeTemplateTitle = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return false;
        }
        const nodeTemplate = node
          ? nodes.nodeTemplates[node.data.type]
          : undefined;

        return nodeTemplate?.title;
      }),
    [nodeId]
  );

  const title = useAppSelector(selector);
  return title;
};
