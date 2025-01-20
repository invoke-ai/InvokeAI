import { useAppDispatch } from 'app/store/storeHooks';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import { useFieldValue } from 'features/nodes/hooks/useFieldValue';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { isEqual } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useFieldDefaultValue = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();

  const value = useFieldValue(nodeId, fieldName);
  const fieldTemplate = useFieldInputTemplate(nodeId, fieldName);

  const isValueChanged = useMemo(() => {
    return !isEqual(value, fieldTemplate.default);
  }, [value, fieldTemplate.default]);

  const resetToDefaultValue = useCallback(() => {
    dispatch(fieldValueReset({ nodeId, fieldName, value: fieldTemplate.default }));
  }, [dispatch, fieldName, fieldTemplate.default, nodeId]);

  return { defaultValue: fieldTemplate.default, isValueChanged, resetToDefaultValue };
};
