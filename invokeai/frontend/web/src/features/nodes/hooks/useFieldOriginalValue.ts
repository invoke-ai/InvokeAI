import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFieldValue } from 'features/nodes/hooks/useFieldValue';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { isEqual } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useFieldOriginalValue = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const selectOriginalExposedFieldValues = useMemo(
    () =>
      createSelector(
        selectWorkflowSlice,
        (workflow) =>
          workflow.originalExposedFieldValues.find((v) => v.nodeId === nodeId && v.fieldName === fieldName)?.value
      ),
    [nodeId, fieldName]
  );
  const originalValue = useAppSelector(selectOriginalExposedFieldValues);
  const value = useFieldValue(nodeId, fieldName);
  const isValueChanged = useMemo(() => !isEqual(value, originalValue), [value, originalValue]);
  const onReset = useCallback(() => {
    dispatch(fieldValueReset({ nodeId, fieldName, value: originalValue }));
  }, [dispatch, fieldName, nodeId, originalValue]);

  return { originalValue, isValueChanged, onReset };
};
