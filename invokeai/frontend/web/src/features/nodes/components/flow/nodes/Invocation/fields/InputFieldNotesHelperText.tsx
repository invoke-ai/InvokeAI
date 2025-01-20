import type { FormHelperTextProps } from '@invoke-ai/ui-library';
import { FormHelperText } from '@invoke-ai/ui-library';
import { useInputFieldNotes } from 'features/nodes/hooks/useInputFieldNotes';
import { memo } from 'react';

type Props = FormHelperTextProps & {
  nodeId: string;
  fieldName: string;
};

export const InputFieldNotesHelperText = memo(({ nodeId, fieldName, ...rest }: Props) => {
  const notes = useInputFieldNotes(nodeId, fieldName);

  if (!notes?.trim()) {
    return null;
  }

  return (
    <FormHelperText px={1} {...rest}>
      {notes}
    </FormHelperText>
  );
});

InputFieldNotesHelperText.displayName = 'InputFieldNotesHelperText';
