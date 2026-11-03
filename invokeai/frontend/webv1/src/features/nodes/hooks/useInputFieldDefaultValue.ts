import { objectEquals } from '@observ33r/object-equals';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { useCallback, useMemo } from 'react';

export const useInputFieldDefaultValue = (fieldName: string) => {
  const dispatch = useAppDispatch();
  const ctx = useInvocationNodeContext();
  const selectDefaultValue = useMemo(
    () => createSelector(ctx.buildSelectInputFieldTemplateOrThrow(fieldName), (fieldTemplate) => fieldTemplate.default),
    [ctx, fieldName]
  );
  const defaultValue = useAppSelector(selectDefaultValue);

  const selectIsChanged = useMemo(
    () =>
      createSelector(
        [ctx.buildSelectInputFieldOrThrow(fieldName), selectDefaultValue],
        (fieldInstance, defaultValue) => {
          return !objectEquals(fieldInstance.value, defaultValue);
        }
      ),
    [fieldName, selectDefaultValue, ctx]
  );
  const isValueChanged = useAppSelector(selectIsChanged);

  const resetToDefaultValue = useCallback(() => {
    dispatch(fieldValueReset({ nodeId: ctx.nodeId, fieldName, value: defaultValue }));
  }, [dispatch, fieldName, defaultValue, ctx.nodeId]);

  return { defaultValue, isValueChanged, resetToDefaultValue };
};
