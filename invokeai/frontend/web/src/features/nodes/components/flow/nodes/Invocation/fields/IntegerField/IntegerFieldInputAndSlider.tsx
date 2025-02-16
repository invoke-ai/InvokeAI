import { CompositeNumberInput, CompositeSlider } from '@invoke-ai/ui-library';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useIntegerField } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/useIntegerField';
import type { IntegerFieldInputInstance, IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { memo } from 'react';

export const IntegerFieldInputAndSlider = memo(
  (props: FieldComponentProps<IntegerFieldInputInstance, IntegerFieldInputTemplate>) => {
    const { defaultValue, onChange, value, min, max, step, fineStep } = useIntegerField(props);

    return (
      <>
        <CompositeSlider
          defaultValue={defaultValue}
          onChange={onChange}
          value={value}
          min={min}
          max={max}
          step={step}
          fineStep={fineStep}
          className="nodrag"
          marks
          withThumbTooltip
          flex="1 1 0"
        />
        <CompositeNumberInput
          defaultValue={defaultValue}
          onChange={onChange}
          value={value}
          min={min}
          max={max}
          step={step}
          fineStep={fineStep}
          className="nodrag"
          flex="1 1 0"
        />
      </>
    );
  }
);

IntegerFieldInputAndSlider.displayName = 'IntegerFieldInputAndSlider';
