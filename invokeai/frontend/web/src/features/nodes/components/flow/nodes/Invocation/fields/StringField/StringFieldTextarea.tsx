import { Textarea } from '@invoke-ai/ui-library';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useStringField } from 'features/nodes/components/flow/nodes/Invocation/fields/StringField/useStringField';
import type { StringFieldInputInstance, StringFieldInputTemplate } from 'features/nodes/types/field';
import { memo } from 'react';

export const StringFieldTextarea = memo(
  (props: FieldComponentProps<StringFieldInputInstance, StringFieldInputTemplate>) => {
    const { value, onChange } = useStringField(props);

    return (
      <Textarea
        className="nodrag nowheel nopan"
        value={value}
        onChange={onChange}
        h="full"
        resize="none"
        fontSize="sm"
        p={2}
      />
    );
  }
);

StringFieldTextarea.displayName = 'StringFieldTextarea';
