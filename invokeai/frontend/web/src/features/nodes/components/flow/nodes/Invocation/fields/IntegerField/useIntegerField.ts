import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { roundDownToMultiple, roundUpToMultiple } from 'common/util/roundDownToMultiple';
import { fieldIntegerValueChanged } from 'features/nodes/store/nodesSlice';
import type { IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { constrainNumber } from 'features/nodes/util/constrainNumber';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';

export const useIntegerField = (
  nodeId: string,
  fieldName: string,
  fieldTemplate: IntegerFieldInputTemplate,
  overrides: { min?: number; max?: number; step?: number } = {}
) => {
  const { min: overrideMin, max: overrideMax, step: overrideStep } = overrides;
  const dispatch = useAppDispatch();

  const step = useMemo(() => {
    if (overrideStep !== undefined) {
      return overrideStep;
    }
    if (isNil(fieldTemplate.multipleOf)) {
      return 1;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf, overrideStep]);

  const fineStep = useMemo(() => {
    if (overrideStep !== undefined) {
      return overrideStep;
    }
    if (isNil(fieldTemplate.multipleOf)) {
      return 1;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf, overrideStep]);

  const min = useMemo(() => {
    let min = -NUMPY_RAND_MAX;

    if (overrideMin !== undefined) {
      min = overrideMin;
    } else if (!isNil(fieldTemplate.minimum)) {
      min = fieldTemplate.minimum;
    } else if (!isNil(fieldTemplate.exclusiveMinimum)) {
      min = fieldTemplate.exclusiveMinimum + 1;
    }

    return roundUpToMultiple(min, step);
  }, [fieldTemplate.exclusiveMinimum, fieldTemplate.minimum, overrideMin, step]);

  const max = useMemo(() => {
    let max = NUMPY_RAND_MAX;

    if (overrideMax !== undefined) {
      max = overrideMax;
    } else if (!isNil(fieldTemplate.maximum)) {
      max = fieldTemplate.maximum;
    } else if (!isNil(fieldTemplate.exclusiveMaximum)) {
      max = fieldTemplate.exclusiveMaximum - 1;
    }

    return roundDownToMultiple(max, step);
  }, [fieldTemplate.exclusiveMaximum, fieldTemplate.maximum, overrideMax, step]);

  const constrainValue = useCallback(
    (v: number) =>
      constrainNumber(v, { min, max, step: step }, { min: overrideMin, max: overrideMax, step: overrideStep }),
    [max, min, overrideMax, overrideMin, overrideStep, step]
  );

  const onChange = useCallback(
    (value: number) => {
      dispatch(fieldIntegerValueChanged({ nodeId, fieldName, value }));
    },
    [dispatch, fieldName, nodeId]
  );

  return {
    defaultValue: fieldTemplate.default,
    onChange,
    min,
    max,
    step,
    fineStep,
    constrainValue,
  };
};
