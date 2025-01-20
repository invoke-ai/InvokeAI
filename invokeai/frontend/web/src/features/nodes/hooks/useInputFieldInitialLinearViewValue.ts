import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInputFieldValue } from 'features/nodes/hooks/useInputFieldValue';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { isEqual } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useInputFieldInitialLinearViewValue = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const selectInitialLinearViewValue = useMemo(
    () =>
      createSelector(
        selectWorkflowSlice,
        (workflow) =>
          workflow.originalExposedFieldValues.find((v) => v.nodeId === nodeId && v.fieldName === fieldName)?.value
      ),
    [nodeId, fieldName]
  );
  const initialLinearViewValue = useAppSelector(selectInitialLinearViewValue);
  const value = useInputFieldValue(nodeId, fieldName);
  const isValueChanged = useMemo(() => !isEqual(value, initialLinearViewValue), [value, initialLinearViewValue]);
  const resetToInitialLinearViewValue = useCallback(() => {
    dispatch(fieldValueReset({ nodeId, fieldName, value: initialLinearViewValue }));
  }, [dispatch, fieldName, nodeId, initialLinearViewValue]);

  return { initialLinearViewValue, isValueChanged, resetToInitialLinearViewValue };
};
