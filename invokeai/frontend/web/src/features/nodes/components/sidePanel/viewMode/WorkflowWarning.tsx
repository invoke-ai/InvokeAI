import { Flex, Icon, Tooltip } from '@invoke-ai/ui-library';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { PiWarningBold } from 'react-icons/pi';

import { WorkflowWarningTooltip } from './WorkflowWarningTooltip';

export const WorkflowWarning = () => {
  const nodesNeedUpdate = useGetNodesNeedUpdate();

  if (!nodesNeedUpdate) {
    return <></>;
  }

  return (
    <Tooltip label={<WorkflowWarningTooltip />}>
      <Flex h="full" alignItems="center" gap="2">
        <Icon color="warning.400" as={PiWarningBold} />
      </Flex>
    </Tooltip>
  );
};
