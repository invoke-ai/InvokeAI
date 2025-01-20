import { Flex, FormLabel, Spacer } from '@invoke-ai/ui-library';
import { InputFieldNotesIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldNotesIconButton';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import InputFieldResetToInitialValueIconButton from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToInitialValueIconButton';
import { useInputFieldLabel } from 'features/nodes/hooks/useInputFieldLabel';
import { useInputFieldTemplateTitle } from 'features/nodes/hooks/useInputFieldTemplateTitle';
import { memo } from 'react';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldViewMode = memo(({ nodeId, fieldName }: Props) => {
  const label = useInputFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitle(nodeId, fieldName);

  return (
    <Flex position="relative" w="full" gap="2" flexDir="column">
      <Flex alignItems="center" gap={1}>
        <FormLabel fontSize="sm">{label || fieldTemplateTitle}</FormLabel>
        <Spacer />
        <InputFieldResetToInitialValueIconButton nodeId={nodeId} fieldName={fieldName} />
        <InputFieldNotesIconButton nodeId={nodeId} fieldName={fieldName} readOnly />
      </Flex>
      <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
    </Flex>
  );
});

InputFieldViewMode.displayName = 'InputFieldViewMode';
