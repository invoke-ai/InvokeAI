import { Flex, Icon, Tooltip } from '@invoke-ai/ui-library';
import { PiWarningBold } from 'react-icons/pi';
import { WorkflowWarningTooltip } from './WorkflowWarningTooltip';
import { useGetNodesNeedUpdate } from '../../../hooks/useGetNodesNeedUpdate';
import { useAppSelector } from '../../../../../app/store/storeHooks';
import { useMemo } from 'react';

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
