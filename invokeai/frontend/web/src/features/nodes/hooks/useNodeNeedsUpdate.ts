import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import { useMemo } from 'react';

export const useNodeNeedsUpdate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        const template = nodes.nodeTemplates[node?.data.type ?? ''];
        return { node, template };
      }),
    [nodeId]
  );

  const { node, template } = useAppSelector(selector);

  const needsUpdate = useMemo(
    () =>
      isInvocationNode(node) && template
        ? getNeedsUpdate(node, template)
        : false,
    [node, template]
  );

  return needsUpdate;
};
