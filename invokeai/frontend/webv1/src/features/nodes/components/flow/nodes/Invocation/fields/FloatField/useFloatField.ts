import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import randomFloat from 'common/util/randomFloat';
import { roundDownToMultiple, roundUpToMultiple } from 'common/util/roundDownToMultiple';
import { isNil } from 'es-toolkit/compat';
import { fieldFloatValueChanged } from 'features/nodes/store/nodesSlice';
import type { FloatFieldInputTemplate } from 'features/nodes/types/field';
import { constrainNumber } from 'features/nodes/util/constrainNumber';
import { useCallback, useMemo } from 'react';

export const useFloatField = (
  nodeId: string,
  fieldName: string,
  fieldTemplate: FloatFieldInputTemplate,
  overrides: { showShuffle?: boolean; min?: number; max?: number; step?: number } = {}
) => {
  const { showShuffle, min: overrideMin, max: overrideMax, step: overrideStep } = overrides;
  const dispatch = useAppDispatch();

  const step = useMemo(() => {
    if (overrideStep !== undefined) {
      return overrideStep;
    }
    if (isNil(fieldTemplate.multipleOf)) {
      return 0.1;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf, overrideStep]);

  const fineStep = useMemo(() => {
    if (overrideStep !== undefined) {
      return overrideStep;
    }
    if (isNil(fieldTemplate.multipleOf)) {
      return 0.01;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf, overrideStep]);

  const baseStep = useMemo(() => {
    if (isNil(fieldTemplate.multipleOf)) {
      return undefined;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf]);

  const min = useMemo(() => {
    let min = -NUMPY_RAND_MAX;

    if (overrideMin !== undefined) {
      min = overrideMin;
    } else if (!isNil(fieldTemplate.minimum)) {
      min = fieldTemplate.minimum;
    } else if (!isNil(fieldTemplate.exclusiveMinimum)) {
      min = fieldTemplate.exclusiveMinimum + 0.01;
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
      max = fieldTemplate.exclusiveMaximum - 0.01;
    }

    return roundDownToMultiple(max, step);
  }, [fieldTemplate.exclusiveMaximum, fieldTemplate.maximum, overrideMax, step]);

  const constrainValue = useCallback(
    (v: number) =>
      constrainNumber(v, { min, max, step: baseStep }, { min: overrideMin, max: overrideMax, step: overrideStep }),
    [max, min, overrideMax, overrideMin, overrideStep, baseStep]
  );

  const onChange = useCallback(
    (value: number) => {
      dispatch(fieldFloatValueChanged({ nodeId, fieldName, value }));
    },
    [dispatch, fieldName, nodeId]
  );

  const randomizeValue = useCallback(() => {
    const value = Number((Math.round(randomFloat(min, max) / step) * step).toFixed(10));
    dispatch(fieldFloatValueChanged({ nodeId, fieldName, value }));
  }, [dispatch, fieldName, nodeId, min, max, step]);

  return {
    defaultValue: fieldTemplate.default,
    onChange,
    min,
    max,
    step,
    fineStep,
    constrainValue,
    showShuffle,
    randomizeValue,
  };
};
