import { Flex, FormLabel, Icon, Spacer, Tooltip } from '@invoke-ai/ui-library';
import FieldTooltipContent from 'features/nodes/components/flow/nodes/Invocation/fields/FieldTooltipContent';
import InputFieldRenderer from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo } from 'react';
import { PiInfoBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

const WorkflowField = ({ nodeId, fieldName }: Props) => {
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, 'input');

  return (
    <Flex layerStyle="second" position="relative" borderRadius="base" w="full" p={4} gap="2" flexDir="column">
      <Flex>
        <FormLabel fontSize="sm">{label || fieldTemplateTitle}</FormLabel>

        <Spacer />
        <Tooltip
          label={<FieldTooltipContent nodeId={nodeId} fieldName={fieldName} kind="input" />}
          openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
          placement="top"
        >
          <Flex h="full" alignItems="center">
            <Icon fontSize="md" color="base.300" as={PiInfoBold} />
          </Flex>
        </Tooltip>
      </Flex>
      <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
    </Flex>
  );
};

export default memo(WorkflowField);
