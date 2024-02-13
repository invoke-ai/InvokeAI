import { CompositeNumberInput } from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldNumberValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  FloatFieldInputInstance,
  FloatFieldInputTemplate,
  IntegerFieldInputInstance,
  IntegerFieldInputTemplate,
} from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';

import type { FieldComponentProps } from './types';

const NumberFieldInputComponent = (
  props: FieldComponentProps<
    IntegerFieldInputInstance | FloatFieldInputInstance,
    IntegerFieldInputTemplate | FloatFieldInputTemplate
  >
) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();
  const isIntegerField = useMemo(() => fieldTemplate.type.name === 'IntegerField', [fieldTemplate.type]);

  const handleValueChanged = useCallback(
    (v: number) => {
      dispatch(
        fieldNumberValueChanged({
          nodeId,
          fieldName: field.name,
          value: isIntegerField ? Math.floor(Number(v)) : Number(v),
        })
      );
    },
    [dispatch, field.name, isIntegerField, nodeId]
  );

  const min = useMemo(() => {
    if (!isNil(fieldTemplate.minimum)) {
      return fieldTemplate.minimum;
    }
    if (!isNil(fieldTemplate.exclusiveMinimum)) {
      return fieldTemplate.exclusiveMinimum + 0.01;
    }
    return;
  }, [fieldTemplate.exclusiveMinimum, fieldTemplate.minimum]);

  const max = useMemo(() => {
    if (!isNil(fieldTemplate.maximum)) {
      return fieldTemplate.maximum;
    }
    if (!isNil(fieldTemplate.exclusiveMaximum)) {
      return fieldTemplate.exclusiveMaximum - 0.01;
    }
    return;
  }, [fieldTemplate.exclusiveMaximum, fieldTemplate.maximum]);

  return (
    <CompositeNumberInput
      defaultValue={fieldTemplate.default}
      onChange={handleValueChanged}
      value={field.value}
      min={min ?? -NUMPY_RAND_MAX}
      max={max ?? NUMPY_RAND_MAX}
      step={isIntegerField ? 1 : 0.1}
      fineStep={isIntegerField ? 1 : 0.01}
      className="nodrag"
    />
  );
};

export default memo(NumberFieldInputComponent);
