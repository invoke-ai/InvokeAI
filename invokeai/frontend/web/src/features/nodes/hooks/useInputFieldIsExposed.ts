import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { useMemo } from 'react';

export const useInputFieldIsExposed = (nodeId: string, fieldName: string) => {
  const selectIsExposed = useMemo(
    () =>
      createSelector(selectWorkflowSlice, (workflow) => {
        return Boolean(workflow.exposedFields.find((f) => f.nodeId === nodeId && f.fieldName === fieldName));
      }),
    [fieldName, nodeId]
  );

  const isExposed = useAppSelector(selectIsExposed);
  return isExposed;
};
