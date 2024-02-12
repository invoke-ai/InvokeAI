import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import LinearViewField from 'features/nodes/components/flow/nodes/Invocation/fields/LinearViewField';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => workflow.exposedFields);

const WorkflowLinearTab = () => {
  const fields = useAppSelector(selector);
  const { isLoading } = useGetOpenAPISchemaQuery();
  const { t } = useTranslation();

  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} h="full" w="full">
          {isLoading ? (
            <IAINoContentFallback label={t('nodes.loadingNodes')} icon={null} />
          ) : fields.length ? (
            fields.map(({ nodeId, fieldName }) => (
              <LinearViewField key={`${nodeId}.${fieldName}`} nodeId={nodeId} fieldName={fieldName} />
            ))
          ) : (
            <IAINoContentFallback label={t('nodes.noFieldsLinearview')} icon={null} />
          )}
        </Flex>
      </ScrollableContent>
    </Box>
  );
};

export default memo(WorkflowLinearTab);
