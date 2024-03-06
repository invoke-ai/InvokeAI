import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { t } from 'i18next';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

import WorkflowField from './WorkflowField';
import { EmptyState } from './EmptyState';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  return {
    fields: workflow.exposedFields,
    name: workflow.name,
  };
});

export const WorkflowViewMode = () => {
  const { isLoading } = useGetOpenAPISchemaQuery();
  const { fields } = useAppSelector(selector);
  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} h="full" w="full">
          {isLoading ? (
            <IAINoContentFallback label={t('nodes.loadingNodes')} icon={null} />
          ) : fields.length ? (
            fields.map(({ nodeId, fieldName }) => (
              <WorkflowField key={`${nodeId}.${fieldName}`} nodeId={nodeId} fieldName={fieldName} />
            ))
          ) : (
            <EmptyState />
          )}
        </Flex>
      </ScrollableContent>
    </Box>
  );
};
