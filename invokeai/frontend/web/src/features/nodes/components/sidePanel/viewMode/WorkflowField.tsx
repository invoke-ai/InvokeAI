import { Flex, FormLabel, Icon, IconButton, Spacer, Tooltip } from '@invoke-ai/ui-library';
import FieldTooltipContent from 'features/nodes/components/flow/nodes/Invocation/fields/FieldTooltipContent';
import InputFieldRenderer from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { InvocationInputFieldCheck } from 'features/nodes/components/flow/nodes/Invocation/fields/InvocationFieldCheck';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldOriginalValue } from 'features/nodes/hooks/useFieldOriginalValue';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { t } from 'i18next';
import { memo } from 'react';
import { PiArrowCounterClockwiseBold, PiInfoBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

const WorkflowFieldInternal = ({ nodeId, fieldName }: Props) => {
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, 'inputs');
  const { isValueChanged, onReset } = useFieldOriginalValue(nodeId, fieldName);

  return (
    <Flex layerStyle="second" position="relative" borderRadius="base" w="full" p={4} gap="2" flexDir="column">
      <Flex alignItems="center">
        <FormLabel fontSize="sm">{label || fieldTemplateTitle}</FormLabel>

        <Spacer />
        {isValueChanged && (
          <IconButton
            aria-label={t('nodes.resetToDefaultValue')}
            tooltip={t('nodes.resetToDefaultValue')}
            variant="ghost"
            size="sm"
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
