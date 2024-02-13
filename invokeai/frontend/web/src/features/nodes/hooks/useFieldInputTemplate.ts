import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldInputTemplate } from 'features/nodes/store/selectors';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldInputTemplate = (nodeId: string, fieldName: string): FieldInputTemplate | null => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        return selectFieldInputTemplate(nodes, nodeId, fieldName);
      }),
    [fieldName, nodeId]
  );

  const fieldTemplate = useAppSelector(selector);

  return fieldTemplate;
};
