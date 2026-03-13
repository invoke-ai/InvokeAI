import { Flex, Icon } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { PiWarningBold } from 'react-icons/pi';

import { WorkflowWarningTooltip } from './WorkflowWarningTooltip';

export const WorkflowWarning = () => {
  const nodesNeedUpdate = useGetNodesNeedUpdate();

  if (!nodesNeedUpdate) {
    return <></>;
  }

  return (
    <IAITooltip label={<WorkflowWarningTooltip />}>
      <Flex h="full" alignItems="center" gap="2">
        <Icon color="warning.400" as={PiWarningBold} />
      </Flex>
    </IAITooltip>
  );
};
