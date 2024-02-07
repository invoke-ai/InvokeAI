import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import QueueControls from 'features/queue/components/QueueControls';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import WorkflowField from './WorkflowField';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => workflow.exposedFields);

const WorkflowPanel = () => {
  const fields = useAppSelector(selector);
  const { t } = useTranslation();

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      <QueueControls />
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
