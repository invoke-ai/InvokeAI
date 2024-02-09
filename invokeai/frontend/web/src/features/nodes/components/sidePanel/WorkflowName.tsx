import { Flex, Icon, Spacer, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { PiInfoBold, PiWarningBold } from 'react-icons/pi';

import WorkflowInfoTooltipContent from './viewMode/WorkflowInfoTooltipContent';
import { WorkflowWarningTooltip } from './workflow/WorkflowWarningTooltip';

export const WorkflowName = () => {
  const name = useAppSelector((s) => s.workflow.name);
  const nodesNeedUpdate = useGetNodesNeedUpdate();

  return (
    <>
      {name.length ? (
        <Flex gap="2" alignItems="center">
          <Tooltip label={<WorkflowInfoTooltipContent />} placement="top">
            <Flex gap="2" alignItems="center">
              <Text fontSize="lg" userSelect="none" noOfLines={1} wordBreak="break-all" fontWeight="semibold">
                {name}
              </Text>

              <Flex h="full" alignItems="center" gap="2">
                <Icon fontSize="lg" color="base.300" as={PiInfoBold} />
              </Flex>
            </Flex>
          </Tooltip>
          {nodesNeedUpdate && (
            <Tooltip label={<WorkflowWarningTooltip />}>
              <Flex h="full" alignItems="center" gap="2">
                <Icon color="warning.400" as={PiWarningBold} />
              </Flex>
            </Tooltip>
          )}
        </Flex>
      ) : (
        <Spacer />
      )}
    </>
  );
};
