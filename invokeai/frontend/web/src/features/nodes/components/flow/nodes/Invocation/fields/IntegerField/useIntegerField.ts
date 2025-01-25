import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { fieldIntegerValueChanged } from 'features/nodes/store/nodesSlice';
import type { IntegerFieldInputInstance, IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useIntegerField = (props: FieldComponentProps<IntegerFieldInputInstance, IntegerFieldInputTemplate>) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();

  const onChange = useCallback(
    (value: number) => {
      dispatch(fieldIntegerValueChanged({ nodeId, fieldName: field.name, value: Math.floor(Number(value)) }));
    },
    [dispatch, field.name, nodeId]
  );

  const min = useMemo(() => {
    let min = -NUMPY_RAND_MAX;
    if (!isNil(fieldTemplate.minimum)) {
      min = fieldTemplate.minimum;
    }
    if (!isNil(fieldTemplate.exclusiveMinimum)) {
      min = fieldTemplate.exclusiveMinimum + 1;
    }
    return min;
  }, [fieldTemplate.exclusiveMinimum, fieldTemplate.minimum]);

  const max = useMemo(() => {
    let max = NUMPY_RAND_MAX;
    if (!isNil(fieldTemplate.maximum)) {
      max = fieldTemplate.maximum;
    }
    if (!isNil(fieldTemplate.exclusiveMaximum)) {
      max = fieldTemplate.exclusiveMaximum - 1;
    }
    return max;
  }, [fieldTemplate.exclusiveMaximum, fieldTemplate.maximum]);

  const step = useMemo(() => {
    if (isNil(fieldTemplate.multipleOf)) {
      return 1;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf]);

  const fineStep = useMemo(() => {
    if (isNil(fieldTemplate.multipleOf)) {
      return 1;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf]);

  return {
    defaultValue: fieldTemplate.default,
    onChange,
    value: field.value,
    min,
    max,
    step,
    fineStep,
  };
};
