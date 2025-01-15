import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFieldInputInstance } from 'features/nodes/hooks/useFieldInputInstance';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { isFloatFieldCollectionInputInstance, isIntegerFieldCollectionInputInstance } from 'features/nodes/types/field';
import { isEqual } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useFieldOriginalValue = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const selectOriginalExposedFieldValues = useMemo(
    () =>
      createMemoizedSelector(selectWorkflowSlice, (workflow) =>
        workflow.originalExposedFieldValues.find((v) => v.nodeId === nodeId && v.fieldName === fieldName)
      ),
    [nodeId, fieldName]
  );
  const exposedField = useAppSelector(selectOriginalExposedFieldValues);
  const field = useFieldInputInstance(nodeId, fieldName);
  const isValueChanged = useMemo(() => {
    if (!field) {
      // Field is not found, so it is not changed
      return false;
    }
    if (isFloatFieldCollectionInputInstance(field) && isFloatFieldCollectionInputInstance(exposedField?.field)) {
      return !isEqual(field.generator, exposedField.field.generator);
    }
    if (isIntegerFieldCollectionInputInstance(field) && isIntegerFieldCollectionInputInstance(exposedField?.field)) {
      return !isEqual(field.generator, exposedField.field.generator);
    }
    return !isEqual(field.value, exposedField?.field.value);
  }, [field, exposedField]);
  const onReset = useCallback(() => {
    if (!exposedField) {
      return;
    }
    const { value } = exposedField.field;
    const generator =
      isIntegerFieldCollectionInputInstance(exposedField.field) ||
      isFloatFieldCollectionInputInstance(exposedField.field)
        ? exposedField.field.generator
        : undefined;
    dispatch(fieldValueReset({ nodeId, fieldName, value, generator }));
  }, [dispatch, fieldName, nodeId, exposedField]);

  return { originalValue: exposedField, isValueChanged, onReset };
};
