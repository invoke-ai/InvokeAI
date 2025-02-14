import { CompositeNumberInput } from '@invoke-ai/ui-library';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useIntegerField } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/useIntegerField';
import type { IntegerFieldInputInstance, IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { memo } from 'react';

export const IntegerFieldInput = memo(
  (props: FieldComponentProps<IntegerFieldInputInstance, IntegerFieldInputTemplate>) => {
    const { defaultValue, onChange, value, min, max, step, fineStep } = useIntegerField(props);

    return (
      <CompositeNumberInput
        defaultValue={defaultValue}
        onChange={onChange}
        value={value}
        min={min}
        max={max}
        step={step}
        fineStep={fineStep}
        className="nodrag"
        flex={1}
      />
    );
  }
);

IntegerFieldInput.displayName = 'IntegerFieldInput';
