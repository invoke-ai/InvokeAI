import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { TEMPLATE_BUILDER_MAP } from 'features/nodes/util/schema/buildFieldInputTemplate';
import { keys, map } from 'lodash-es';
import { useMemo } from 'react';

export const useAnyOrDirectInputFieldNames = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes }) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return [];
        }
        const nodeTemplate = nodes.nodeTemplates[node.data.type];
        if (!nodeTemplate) {
          return [];
        }
        const fields = map(nodeTemplate.inputs).filter(
          (field) =>
            (['any', 'direct'].includes(field.input) ||
              field.type.isCollectionOrScalar) &&
            keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
        );
        return getSortedFilteredFieldNames(fields);
      }),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
