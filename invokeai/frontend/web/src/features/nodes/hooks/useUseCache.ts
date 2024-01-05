import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useUseCache = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return false;
        }
        // cast to boolean to support older workflows that didn't have useCache
        // TODO: handle this better somehow
        return node.data.useCache;
      }),
    [nodeId]
  );

  const useCache = useAppSelector(selector);
  return useCache;
};
