import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldInputInstance } from 'features/nodes/store/selectors';
import type { FieldInputInstance } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldInputInstance = (nodeId: string, fieldName: string): FieldInputInstance | null => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        return selectFieldInputInstance(nodes, nodeId, fieldName);
      }),
    [fieldName, nodeId]
  );

  const fieldData = useAppSelector(selector);

  return fieldData;
};
