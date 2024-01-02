import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { map } from 'lodash-es';
import { useMemo } from 'react';

export const useOutputFieldNames = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes, nodeTemplates }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return [];
        }
        const nodeTemplate = nodeTemplates.templates[node.data.type];
        if (!nodeTemplate) {
          return [];
        }

        return getSortedFilteredFieldNames(map(nodeTemplate.outputs));
      }),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
