import { useNode } from 'features/nodes/hooks/useNode';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useBatchGroupId = (nodeId: string) => {
  const node = useNode(nodeId);

  const batchGroupId = useMemo(() => {
    if (!isInvocationNode(node)) {
      return;
    }
    if (!isBatchNode(node)) {
      return;
    }
    return node.data.inputs['batch_group_id']?.value as string;
  }, [node]);

  return batchGroupId;
};
