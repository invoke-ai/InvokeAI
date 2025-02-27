import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectFieldInputInstanceSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useInputFieldLabel = (nodeId: string, fieldName: string): string => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectFieldInputInstanceSafe(nodes, nodeId, fieldName)?.label ?? '';
      }),
    [fieldName, nodeId]
  );

  const label = useAppSelector(selector);

  return label;
};
