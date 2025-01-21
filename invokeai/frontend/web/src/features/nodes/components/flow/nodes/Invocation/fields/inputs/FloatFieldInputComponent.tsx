import { CompositeNumberInput } from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldFloatValueChanged } from 'features/nodes/store/nodesSlice';
import type { FloatFieldInputInstance, FloatFieldInputTemplate } from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';

import type { FieldComponentProps } from './types';

export const FloatFieldInputComponent = memo(
  (props: FieldComponentProps<FloatFieldInputInstance, FloatFieldInputTemplate>) => {
    const { nodeId, field, fieldTemplate } = props;
    const dispatch = useAppDispatch();

    const handleValueChanged = useCallback(
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

    return (
      <CompositeNumberInput
        defaultValue={fieldTemplate.default}
        onChange={handleValueChanged}
        value={field.value}
        min={min}
        max={max}
        step={step}
        fineStep={fineStep}
        className="nodrag"
      />
    );
  }
);

FloatFieldInputComponent.displayName = 'FloatFieldInputComponent ';
