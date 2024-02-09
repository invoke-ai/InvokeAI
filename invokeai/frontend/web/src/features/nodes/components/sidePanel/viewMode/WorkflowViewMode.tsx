import { t } from 'i18next';
import { createMemoizedSelector } from '../../../../../app/store/createMemoizedSelector';
import { useAppSelector } from '../../../../../app/store/storeHooks';
import { IAINoContentFallback } from '../../../../../common/components/IAIImageFallback';
import ScrollableContent from '../../../../../common/components/OverlayScrollbars/ScrollableContent';
import { selectWorkflowSlice } from '../../../store/workflowSlice';
import WorkflowField from './WorkflowField';
import { Box, Flex } from '@invoke-ai/ui-library';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  return {
    fields: workflow.exposedFields,
    name: workflow.name,
  };
});

export const WorkflowViewMode = () => {
  const { fields } = useAppSelector(selector);
  return (
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
  );
};
