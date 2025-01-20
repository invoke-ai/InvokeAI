import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldViewSimple } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldViewSimple';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { t } from 'i18next';
import { memo } from 'react';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

import { EmptyState } from './EmptyState';

const selectExposedFields = createMemoizedSelector(selectWorkflowSlice, (workflow) => workflow.exposedFields);

export const FieldsSimpleView = memo(() => {
  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} w="full" h="full">
          <FieldSimpleViewContent />
        </Flex>
      </ScrollableContent>
    </Box>
  );
});
FieldsSimpleView.displayName = 'FieldsSimpleView';

const FieldSimpleViewContent = memo(() => {
  const { isLoading } = useGetOpenAPISchemaQuery();
  const exposedFields = useAppSelector(selectExposedFields);

  if (isLoading) {
    return <IAINoContentFallback label={t('nodes.loadingNodes')} icon={null} />;
  }

  if (exposedFields.length === 0) {
    return <EmptyState />;
  }

  return (
    <>
      {exposedFields.map(({ nodeId, fieldName }) => (
        <InputFieldGate key={`${nodeId}.${fieldName}`} nodeId={nodeId} fieldName={fieldName}>
          <InputFieldViewSimple nodeId={nodeId} fieldName={fieldName} />
        </InputFieldGate>
      ))}
    </>
  );
});
FieldSimpleViewContent.displayName = 'FieldSimpleViewContent';
