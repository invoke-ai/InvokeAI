import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectFieldInputInstanceSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import type { FieldInputInstance } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useInputFieldInstance = (nodeId: string, fieldName: string): FieldInputInstance => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const instance = selectFieldInputInstanceSafe(nodes, nodeId, fieldName);
        assert(instance, `Instance for input field ${fieldName} not found`);
        return instance;
      }),
    [fieldName, nodeId]
  );

  const instance = useAppSelector(selector);

  return instance;
};
