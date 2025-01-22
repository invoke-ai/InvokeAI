import { Box, FormControl, FormLabel, Spacer } from '@invoke-ai/ui-library';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { InputFieldResetToInitialValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToInitialValueIconButton';
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
    <FormControl w="full" gap={2} flexDir="column">
      <FormLabel fontSize="sm" display="flex" w="full" m={0} gap={2} ps={1}>
        {label || fieldTemplateTitle}
        <Spacer />
        <InputFieldResetToInitialValueIconButton nodeId={nodeId} fieldName={fieldName} />
      </FormLabel>
      <Box w="full" h="full">
        <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
      </Box>
    </FormControl>
  );
});

InputFieldViewMode.displayName = 'InputFieldViewMode';
