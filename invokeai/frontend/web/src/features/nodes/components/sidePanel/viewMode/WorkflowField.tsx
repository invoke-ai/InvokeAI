import { Flex, FormLabel, IconButton, Spacer } from '@invoke-ai/ui-library';
import { FieldNotesIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldNotesIconButton';
import InputFieldRenderer from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { InvocationInputFieldCheck } from 'features/nodes/components/flow/nodes/Invocation/fields/InvocationFieldCheck';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldOriginalValue } from 'features/nodes/hooks/useFieldOriginalValue';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import { t } from 'i18next';
import { memo } from 'react';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

const WorkflowFieldInternal = ({ nodeId, fieldName }: Props) => {
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, 'inputs');
  const { isValueChanged, onReset } = useFieldOriginalValue(nodeId, fieldName);

  return (
    <Flex position="relative" w="full" gap="2" flexDir="column">
      <Flex alignItems="center" gap={1}>
        <FormLabel fontSize="sm">{label || fieldTemplateTitle}</FormLabel>
        <Spacer />
        {isValueChanged && (
          <IconButton
            aria-label={t('nodes.resetToDefaultValue')}
            tooltip={t('nodes.resetToDefaultValue')}
            variant="ghost"
            size="xs"
            onClick={onReset}
            icon={<PiArrowCounterClockwiseBold />}
          />
        )}
        <FieldNotesIconButton nodeId={nodeId} fieldName={fieldName} readOnly />
      </Flex>
      <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
    </Flex>
  );
};

const WorkflowField = ({ nodeId, fieldName }: Props) => {
  return (
    <InvocationInputFieldCheck nodeId={nodeId} fieldName={fieldName}>
      <WorkflowFieldInternal nodeId={nodeId} fieldName={fieldName} />
    </InvocationInputFieldCheck>
  );
};

export default memo(WorkflowField);
