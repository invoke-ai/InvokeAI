import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { isEqual } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useInputFieldDefaultValue = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();

  const fieldTemplate = useInputFieldTemplate(nodeId, fieldName);
  const selectIsChanged = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return;
        }
        const value = node.data.inputs[fieldName]?.value;
        return !isEqual(value, fieldTemplate.default);
      }),
    [fieldName, fieldTemplate.default, nodeId]
  );
  const isValueChanged = useAppSelector(selectIsChanged);

  const resetToDefaultValue = useCallback(() => {
    dispatch(fieldValueReset({ nodeId, fieldName, value: fieldTemplate.default }));
  }, [dispatch, fieldName, fieldTemplate.default, nodeId]);

  return { defaultValue: fieldTemplate.default, isValueChanged, resetToDefaultValue };
};
