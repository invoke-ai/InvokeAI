import { Box, Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import QueueControls from 'features/queue/components/QueueControls';
import WorkflowLibraryButton from 'features/workflowLibrary/components/WorkflowLibraryButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

import WorkflowField from './WorkflowField';
import WorkflowInfoTooltipContent from './WorkflowInfoTooltipContent';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  return {
    fields: workflow.exposedFields,
    name: workflow.name,
  };
});

const WorkflowPanel = () => {
  const { fields, name } = useAppSelector(selector);
  const { t } = useTranslation();

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      <QueueControls />
      <Flex w="full" justifyContent="space-between" alignItems="center" gap="4" padding={1}>
        <Tooltip label={<WorkflowInfoTooltipContent />} placement="top">
          <Flex gap="2" alignItems="center">
            <Text fontSize="lg" userSelect="none" noOfLines={1} wordBreak="break-all" fontWeight="semibold">
              {name}
            </Text>

            <Flex h="full" alignItems="center">
              <Icon fontSize="lg" color="base.300" as={PiInfoBold} />
            </Flex>
          </Flex>
        </Tooltip>
        <Flex>
          <WorkflowLibraryButton />
        </Flex>
      </Flex>
      <Box position="relative" w="full" h="full">
        <ScrollableContent>
          <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} h="full" w="full">
            {fields.length ? (
              fields.map(({ nodeId, fieldName }) => (
                <WorkflowField key={`${nodeId}.${fieldName}`} nodeId={nodeId} fieldName={fieldName} />
              ))
            ) : (
              <IAINoContentFallback label={t('nodes.noFieldsLinearview')} icon={null} />
            )}
          </Flex>
        </ScrollableContent>
      </Box>
    </Flex>
  );
};

export default memo(WorkflowPanel);
