import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldInputTemplate, selectFieldOutputTemplate } from 'features/nodes/store/selectors';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldTemplate = (
  nodeId: string,
  fieldName: string,
  kind: 'inputs' | 'outputs'
): FieldInputTemplate | FieldOutputTemplate | null => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        if (kind === 'inputs') {
          return selectFieldInputTemplate(nodes, nodeId, fieldName);
        }
        return selectFieldOutputTemplate(nodes, nodeId, fieldName);
      }),
    [fieldName, kind, nodeId]
  );

  const fieldTemplate = useAppSelector(selector);

  return fieldTemplate;
};
