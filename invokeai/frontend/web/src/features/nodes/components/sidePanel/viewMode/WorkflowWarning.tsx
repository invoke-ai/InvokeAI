import { Flex, Icon, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { useMemo } from 'react';
import { PiWarningBold } from 'react-icons/pi';

import { WorkflowWarningTooltip } from './WorkflowWarningTooltip';

export const WorkflowWarning = () => {
  const nodesNeedUpdate = useGetNodesNeedUpdate();
  const isTouched = useAppSelector((s) => s.workflow.isTouched);

  const showWarning = useMemo(() => {
    return nodesNeedUpdate || isTouched;
  }, [nodesNeedUpdate, isTouched]);

  if (!showWarning) {
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
