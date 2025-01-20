import { FormControl, FormLabel, Spacer } from '@invoke-ai/ui-library';
import { InputFieldNotesHelperText } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldNotesHelperText';
import { InputFieldNotesIconButtonReadonly } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldNotesIconButtonReadonly';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { InputFieldResetToInitialValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToInitialValueIconButton';
import { useInputFieldLabel } from 'features/nodes/hooks/useInputFieldLabel';
import { useInputFieldLinearViewConfig } from 'features/nodes/hooks/useInputFieldLinearViewConfig';
import { useInputFieldTemplateTitle } from 'features/nodes/hooks/useInputFieldTemplateTitle';
import { memo } from 'react';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldViewMode = memo(({ nodeId, fieldName }: Props) => {
  const label = useInputFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitle(nodeId, fieldName);
  const linearViewConfig = useInputFieldLinearViewConfig(nodeId, fieldName);

  return (
    <FormControl w="full" gap={2} flexDir="column">
      <FormLabel fontSize="sm" display="flex" w="full" m={0} gap={2} px={1}>
        {label || fieldTemplateTitle}
        <Spacer />
        {linearViewConfig?.notesDisplay === 'icon-with-popover' && (
          <InputFieldNotesIconButtonReadonly nodeId={nodeId} fieldName={fieldName} />
        )}
        <InputFieldResetToInitialValueIconButton nodeId={nodeId} fieldName={fieldName} />
      </FormLabel>
      <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
      {linearViewConfig?.notesDisplay === 'helper-text' && (
        <InputFieldNotesHelperText nodeId={nodeId} fieldName={fieldName} />
      )}
    </FormControl>
  );
});

InputFieldViewMode.displayName = 'InputFieldViewMode';
