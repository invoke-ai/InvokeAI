import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { EMPTY_ARRAY } from 'app/store/util';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplate } from 'features/nodes/store/selectors';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { TEMPLATE_BUILDER_MAP } from 'features/nodes/util/schema/buildFieldInputTemplate';
import { keys, map } from 'lodash-es';
import { useMemo } from 'react';

export const useAnyOrDirectInputFieldNames = (nodeId: string): string[] => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const template = selectNodeTemplate(nodes, nodeId);
        if (!template) {
          return EMPTY_ARRAY;
        }
        const fields = map(template.inputs).filter(
          (field) =>
            (['any', 'direct'].includes(field.input) || field.type.isCollectionOrScalar) &&
            keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
        );
        return getSortedFilteredFieldNames(fields);
      }),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
