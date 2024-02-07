import { Flex, Icon, Spacer, Tooltip } from '@invoke-ai/ui-library';
import FieldTitle from 'features/nodes/components/flow/nodes/Invocation/fields/FieldTitle';
import FieldTooltipContent from 'features/nodes/components/flow/nodes/Invocation/fields/FieldTooltipContent';
import InputFieldRenderer from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo } from 'react';
import { PiInfoBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

const WorkflowField = ({ nodeId, fieldName }: Props) => {
  return (
    <Flex layerStyle="second" position="relative" borderRadius="base" w="full" p={4} flexDir="column">
      <Flex>
        <FieldTitle nodeId={nodeId} fieldName={fieldName} kind="input" />
        <Spacer />
        <Tooltip
          label={<FieldTooltipContent nodeId={nodeId} fieldName={fieldName} kind="input" />}
          openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
          placement="top"
        >
          <Flex h="full" alignItems="center">
            <Icon fontSize="sm" color="base.300" as={PiInfoBold} />
          </Flex>
        </Tooltip>
      </Flex>
      <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
    </Flex>
  );
};

export default memo(WorkflowField);
