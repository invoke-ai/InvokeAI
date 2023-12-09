import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';

export const useNodeData = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        return node?.data;
      }),
    [nodeId]
  );

  const nodeData = useAppSelector(selector);

  return nodeData;
};
