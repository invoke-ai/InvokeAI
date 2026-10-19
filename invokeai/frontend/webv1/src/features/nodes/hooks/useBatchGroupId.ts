import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useBatchGroupId = (nodeId: string) => {
  const selector = useMemo(() => {
    return createSelector(selectNodesSlice, (nodes) => {
      const node = selectNode(nodes, nodeId);
      if (!isInvocationNode(node)) {
        return;
      }
      if (!isBatchNode(node)) {
        return;
      }
      return node.data.inputs['batch_group_id']?.value as string;
    });
  }, [nodeId]);

  const batchGroupId = useAppSelector(selector);

  return batchGroupId;
};
