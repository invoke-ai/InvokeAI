import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectFieldInputInstance, selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useInputFieldLabel = (nodeId: string, fieldName: string): string => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        return selectFieldInputInstance(nodes, nodeId, fieldName)?.label ?? '';
      }),
    [fieldName, nodeId]
  );

  const label = useAppSelector(selector);

  return label;
};
