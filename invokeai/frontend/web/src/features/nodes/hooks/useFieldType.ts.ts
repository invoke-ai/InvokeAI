import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { KIND_MAP } from 'features/nodes/types/constants';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useFieldType = (nodeId: string, fieldName: string, kind: 'input' | 'output') => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return;
        }
        const field = node.data[KIND_MAP[kind]][fieldName];
        return field?.type;
      }),
    [fieldName, kind, nodeId]
  );

  const fieldType = useAppSelector(selector);

  return fieldType;
};
