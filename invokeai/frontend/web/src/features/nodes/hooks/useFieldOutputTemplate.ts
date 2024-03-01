import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldOutputTemplate } from 'features/nodes/store/selectors';
import type { FieldOutputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldOutputTemplate = (nodeId: string, fieldName: string): FieldOutputTemplate | null => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        return selectFieldOutputTemplate(nodes, nodeId, fieldName);
      }),
    [fieldName, nodeId]
  );

  const fieldTemplate = useAppSelector(selector);

  return fieldTemplate;
};
