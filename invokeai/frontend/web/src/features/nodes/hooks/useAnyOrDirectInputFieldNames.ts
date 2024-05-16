import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { TEMPLATE_BUILDER_MAP } from 'features/nodes/util/schema/buildFieldInputTemplate';
import { keys, map } from 'lodash-es';
import { useMemo } from 'react';

export const useAnyOrDirectInputFieldNames = (nodeId: string): string[] => {
  const template = useNodeTemplate(nodeId);
  const selectConnectedFieldNames = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodesSlice) =>
        nodesSlice.edges
          .filter((e) => e.target === nodeId)
          .map((e) => e.targetHandle)
          .filter(Boolean)
      ),
    [nodeId]
  );
  const connectedFieldNames = useAppSelector(selectConnectedFieldNames);

  const fieldNames = useMemo(() => {
    const fields = map(template.inputs).filter((field) => {
      if (connectedFieldNames.includes(field.name)) {
        return false;
      }

      return (
        (['any', 'direct'].includes(field.input) || field.type.isCollectionOrScalar) &&
        keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
      );
    });
    return getSortedFilteredFieldNames(fields);
  }, [connectedFieldNames, template.inputs]);

  return fieldNames;
};
