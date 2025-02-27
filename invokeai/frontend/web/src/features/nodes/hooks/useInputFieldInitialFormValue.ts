import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
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
  const selectIsChanged = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        if (initialValue === uniqueNonexistentValue) {
          return false;
        }
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return;
        }
        const value = node.data.inputs[fieldName]?.value;
        return !isEqual(value, initialValue);
      }),
    [fieldName, initialValue, nodeId]
  );
  const isValueChanged = useAppSelector(selectIsChanged);
  const resetToInitialValue = useCallback(() => {
    if (initialValue === uniqueNonexistentValue) {
      return;
    }
    dispatch(fieldValueReset({ nodeId, fieldName, value: initialValue }));
  }, [dispatch, fieldName, nodeId, initialValue]);

  return { initialValue, isValueChanged, resetToInitialValue };
};
