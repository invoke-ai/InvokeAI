import { Flex, FormLabel, Spacer } from '@invoke-ai/ui-library';
import { FieldNotesIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldNotesIconButton';
import FieldResetToInitialLinearViewValueButton from 'features/nodes/components/flow/nodes/Invocation/fields/FieldResetToInitialLinearViewValueButton';
import InputFieldRenderer from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import { memo } from 'react';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldViewSimple = memo(({ nodeId, fieldName }: Props) => {
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, 'inputs');

  return (
    <Flex position="relative" w="full" gap="2" flexDir="column">
      <Flex alignItems="center" gap={1}>
        <FormLabel fontSize="sm">{label || fieldTemplateTitle}</FormLabel>
        <Spacer />
        <FieldResetToInitialLinearViewValueButton nodeId={nodeId} fieldName={fieldName} />
        <FieldNotesIconButton nodeId={nodeId} fieldName={fieldName} readOnly />
      </Flex>
      <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
    </Flex>
  );
});

InputFieldViewSimple.displayName = 'InputFieldViewSimple';
