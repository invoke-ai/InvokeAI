import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldInputTemplate } from 'features/nodes/store/selectors';
import type { FieldInput } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldInputKind = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes): FieldInput | null => {
        const template = selectFieldInputTemplate(nodes, nodeId, fieldName);
        return template?.input ?? null;
      }),
    [fieldName, nodeId]
  );

  const inputKind = useAppSelector(selector);

  return inputKind;
};
