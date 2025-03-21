import { Select } from '@invoke-ai/ui-library';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useStringField } from 'features/nodes/components/flow/nodes/Invocation/fields/StringField/useStringField';
import { NO_DRAG_CLASS, NO_PAN_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { StringFieldInputInstance, StringFieldInputTemplate } from 'features/nodes/types/field';
import type { NodeFieldStringSettings } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const StringFieldDropdown = memo(
  (
    props: FieldComponentProps<
      StringFieldInputInstance,
      StringFieldInputTemplate,
      { settings: Extract<NodeFieldStringSettings, { component: 'dropdown' }> }
    >
  ) => {
    const { value, onChange } = useStringField(props);

    return (
      <Select
        onChange={onChange}
        className={`${NO_DRAG_CLASS} ${NO_PAN_CLASS} ${NO_WHEEL_CLASS}`}
        isDisabled={props.settings.options.length === 0}
        value={value}
      >
        {props.settings.options.map((choice, i) => (
          <option key={`${i}_${choice.value}`} value={choice.value}>
            {choice.label || choice.value || `Option ${i + 1}`}
          </option>
        ))}
      </Select>
    );
  }
);

StringFieldDropdown.displayName = 'StringFieldDropdown';
