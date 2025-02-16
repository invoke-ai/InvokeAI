import { useAppDispatch } from 'app/store/storeHooks';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { useInputFieldValue } from 'features/nodes/hooks/useInputFieldValue';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { isEqual } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useInputFieldDefaultValue = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();

  const value = useInputFieldValue(nodeId, fieldName);
  const fieldTemplate = useInputFieldTemplate(nodeId, fieldName);

  const isValueChanged = useMemo(() => {
    return !isEqual(value, fieldTemplate.default);
  }, [value, fieldTemplate.default]);

  const resetToDefaultValue = useCallback(() => {
    dispatch(fieldValueReset({ nodeId, fieldName, value: fieldTemplate.default }));
  }, [dispatch, fieldName, fieldTemplate.default, nodeId]);

  return { defaultValue: fieldTemplate.default, isValueChanged, resetToDefaultValue };
};
