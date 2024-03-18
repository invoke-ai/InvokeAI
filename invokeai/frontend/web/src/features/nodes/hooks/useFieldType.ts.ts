import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldInputTemplate, selectFieldOutputTemplate } from 'features/nodes/store/selectors';
import type { FieldType } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldType = (nodeId: string, fieldName: string, kind: 'inputs' | 'outputs'): FieldType | null => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        if (kind === 'inputs') {
          return selectFieldInputTemplate(nodes, nodeId, fieldName)?.type ?? null;
        }
        return selectFieldOutputTemplate(nodes, nodeId, fieldName)?.type ?? null;
      }),
    [fieldName, kind, nodeId]
  );

  const fieldType = useAppSelector(selector);

  return fieldType;
};
