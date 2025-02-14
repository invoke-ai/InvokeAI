import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectFieldInputInstance, selectNodesSlice } from 'features/nodes/store/selectors';
import type { FieldInputInstance } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useInputFieldInstance = (nodeId: string, fieldName: string): FieldInputInstance => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const instance = selectFieldInputInstance(nodes, nodeId, fieldName);
        assert(instance, `Instance for input field ${fieldName} not found`);
        return instance;
      }),
    [fieldName, nodeId]
  );

  const instance = useAppSelector(selector);

  return instance;
};
