import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInputFieldValue } from 'features/nodes/hooks/useInputFieldValue';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { isEqual } from 'lodash-es';
import { useCallback, useMemo } from 'react';

const uniqueNonexistentValue = Symbol('uniqueNonexistentValue');

export const useInputFieldInitialFormValue = (elementId: string, nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const selectInitialValue = useMemo(
    () =>
      createSelector(selectWorkflowSlice, (workflow) => {
        if (!(elementId in workflow.formFieldInitialValues)) {
          return uniqueNonexistentValue;
        }
        return workflow.formFieldInitialValues[elementId];
      }),
    [elementId]
  );
  const initialValue = useAppSelector(selectInitialValue);
  const value = useInputFieldValue(nodeId, fieldName);
  const isValueChanged = useMemo(
    () => initialValue !== uniqueNonexistentValue && !isEqual(value, initialValue),
    [value, initialValue]
  );
  const resetToInitialValue = useCallback(() => {
    if (initialValue === uniqueNonexistentValue) {
      return;
    }
    dispatch(fieldValueReset({ nodeId, fieldName, value: initialValue }));
  }, [dispatch, fieldName, nodeId, initialValue]);

  return { initialValue, isValueChanged, resetToInitialValue };
};
