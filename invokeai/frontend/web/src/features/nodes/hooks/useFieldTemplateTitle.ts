import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldInputTemplate, selectFieldOutputTemplate } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useFieldTemplateTitle = (nodeId: string, fieldName: string, kind: 'inputs' | 'outputs'): string | null => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        if (kind === 'inputs') {
          return selectFieldInputTemplate(nodes, nodeId, fieldName)?.title ?? null;
        }
        return selectFieldOutputTemplate(nodes, nodeId, fieldName)?.title ?? null;
      }),
    [fieldName, kind, nodeId]
  );

  const fieldTemplateTitle = useAppSelector(selector);

  return fieldTemplateTitle;
};
