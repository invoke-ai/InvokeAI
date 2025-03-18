import { CompositeNumberInput } from '@invoke-ai/ui-library';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useIntegerField } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/useIntegerField';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { IntegerFieldInputInstance, IntegerFieldInputTemplate } from 'features/nodes/types/field';
import type { NodeFieldIntegerSettings } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const IntegerFieldInput = memo(
  (
    props: FieldComponentProps<
      IntegerFieldInputInstance,
      IntegerFieldInputTemplate,
      { settings?: NodeFieldIntegerSettings }
    >
  ) => {
    const { nodeId, field, fieldTemplate, settings } = props;
    const { defaultValue, onChange, min, max, step, fineStep, constrainValue } = useIntegerField(
      nodeId,
      field.name,
      fieldTemplate,
      settings
    );

    return (
      <CompositeNumberInput
        defaultValue={defaultValue}
        onChange={onChange}
        value={field.value}
        min={min}
        max={max}
        step={step}
        fineStep={fineStep}
        className={NO_DRAG_CLASS}
        flex="1 1 0"
        constrainValue={constrainValue}
      />
    );
  }
);

IntegerFieldInput.displayName = 'IntegerFieldInput';
