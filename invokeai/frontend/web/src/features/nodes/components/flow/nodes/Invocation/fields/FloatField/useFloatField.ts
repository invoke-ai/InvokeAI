import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { fieldFloatValueChanged } from 'features/nodes/store/nodesSlice';
import type { FloatFieldInputInstance, FloatFieldInputTemplate } from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useFloatField = (props: FieldComponentProps<FloatFieldInputInstance, FloatFieldInputTemplate>) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();

  const onChange = useCallback(
    (value: number) => {
      dispatch(fieldFloatValueChanged({ nodeId, fieldName: field.name, value }));
    },
    [dispatch, field.name, nodeId]
  );

  const min = useMemo(() => {
    let min = -NUMPY_RAND_MAX;
    if (!isNil(fieldTemplate.minimum)) {
      min = fieldTemplate.minimum;
    }
    if (!isNil(fieldTemplate.exclusiveMinimum)) {
      min = fieldTemplate.exclusiveMinimum + 0.01;
    }
    return min;
  }, [fieldTemplate.exclusiveMinimum, fieldTemplate.minimum]);

  const max = useMemo(() => {
    let max = NUMPY_RAND_MAX;
    if (!isNil(fieldTemplate.maximum)) {
      max = fieldTemplate.maximum;
    }
    if (!isNil(fieldTemplate.exclusiveMaximum)) {
      max = fieldTemplate.exclusiveMaximum - 0.01;
    }
    return max;
  }, [fieldTemplate.exclusiveMaximum, fieldTemplate.maximum]);

  const step = useMemo(() => {
    if (isNil(fieldTemplate.multipleOf)) {
      return 0.1;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf]);

  const fineStep = useMemo(() => {
    if (isNil(fieldTemplate.multipleOf)) {
      return 0.01;
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
