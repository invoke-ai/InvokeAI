import { EMPTY_ARRAY } from "app/store/constants";
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplate } from 'features/nodes/store/selectors';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { TEMPLATE_BUILDER_MAP } from 'features/nodes/util/schema/buildFieldInputTemplate';
import { keys, map } from 'lodash-es';
import { useMemo } from 'react';

export const useConnectionInputFieldNames = (nodeId: string): string[] => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const template = selectNodeTemplate(nodes, nodeId);
        if (!template) {
          return EMPTY_ARRAY;
        }

        // get the visible fields
        const fields = map(template.inputs).filter(
          (field) =>
            (field.input === 'connection' && !field.type.isCollectionOrScalar) ||
            !keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
        );

        return getSortedFilteredFieldNames(fields);
      }),
    [nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};
