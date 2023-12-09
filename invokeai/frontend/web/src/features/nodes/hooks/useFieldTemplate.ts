import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { KIND_MAP } from 'features/nodes/types/constants';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

export const useFieldTemplate = (
  nodeId: string,
  fieldName: string,
  kind: 'input' | 'output'
) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return;
        }
        const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];
        return nodeTemplate?.[KIND_MAP[kind]][fieldName];
      }),
    [fieldName, kind, nodeId]
  );

  const fieldTemplate = useAppSelector(selector);

  return fieldTemplate;
};
