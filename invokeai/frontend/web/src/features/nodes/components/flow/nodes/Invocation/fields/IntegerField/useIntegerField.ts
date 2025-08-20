import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import randomInt from 'common/util/randomInt';
import { roundDownToMultiple, roundUpToMultiple } from 'common/util/roundDownToMultiple';
import { isNil } from 'es-toolkit/compat';
import { fieldIntegerValueChanged } from 'features/nodes/store/nodesSlice';
import type { IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { constrainNumber } from 'features/nodes/util/constrainNumber';
import { useCallback, useMemo } from 'react';

export const useIntegerField = (
  nodeId: string,
  fieldName: string,
  fieldTemplate: IntegerFieldInputTemplate,
  overrides: { showShuffle?: boolean; min?: number; max?: number; step?: number } = {}
) => {
  const { showShuffle, min: overrideMin, max: overrideMax, step: overrideStep } = overrides;
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
    (v: number) => constrainNumber(v, { min, max, step }, { min: overrideMin, max: overrideMax, step: overrideStep }),
    [max, min, overrideMax, overrideMin, overrideStep, step]
  );

  const onValueChange = useCallback(
    (value: number) => {
      dispatch(fieldIntegerValueChanged({ nodeId, fieldName, value }));
    },
    [dispatch, fieldName, nodeId]
  );

  const handleClickRandomizeValue = useCallback(() => {
    const value = Math.round(randomInt(min, max) / step) * step;
    dispatch(fieldIntegerValueChanged({ nodeId, fieldName, value }));
  }, [dispatch, fieldName, nodeId, min, max, step]);

  return {
    defaultValue: fieldTemplate.default,
    onValueChange,
    min,
    max,
    step,
    fineStep,
    constrainValue,
    showShuffle,
    handleClickRandomizeValue,
  };
};
