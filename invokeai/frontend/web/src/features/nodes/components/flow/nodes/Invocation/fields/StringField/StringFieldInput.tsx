import { Input } from '@invoke-ai/ui-library';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useStringField } from 'features/nodes/components/flow/nodes/Invocation/fields/StringField/useStringField';
import type { StringFieldInputInstance, StringFieldInputTemplate } from 'features/nodes/types/field';
import { memo } from 'react';

export const StringFieldInput = memo(
  (props: FieldComponentProps<StringFieldInputInstance, StringFieldInputTemplate>) => {
    const { value, onChange } = useStringField(props);

    return <Input className="nodrag nowheel nopan" value={value} onChange={onChange} />;
  }
);

StringFieldInput.displayName = 'StringFieldInput';
