import { DndContext } from '@dnd-kit/core';
import { arrayMove, SortableContext } from '@dnd-kit/sortable';
import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { DragEndEvent } from 'features/dnd/types';
import LinearViewField from 'features/nodes/components/flow/nodes/Invocation/fields/LinearViewField';
import { selectWorkflowSlice, workflowExposedFieldsReordered } from 'features/nodes/store/workflowSlice';
import { FieldIdentifier } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => workflow.exposedFields);

const WorkflowLinearTab = () => {
  const fields = useAppSelector(selector);
  const { isLoading } = useGetOpenAPISchemaQuery();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      console.log({ active, over });
      const fieldsStrings = fields.map((field) => `${field.nodeId}.${field.fieldName}`);

      if (over && active.id !== over.id) {
        const oldIndex = fieldsStrings.indexOf(active.id as string);
        const newIndex = fieldsStrings.indexOf(over.id as string);

        const newFields = arrayMove(fieldsStrings, oldIndex, newIndex)
          .map((field) => fields.find((obj) => `${obj.nodeId}.${obj.fieldName}` === field))
          .filter((field) => field) as FieldIdentifier[];

        dispatch(workflowExposedFieldsReordered(newFields));
      }
    },
    [dispatch, fields]
  );

  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <DndContext onDragEnd={handleDragEnd}>
          <SortableContext items={fields.map((field) => `${field.nodeId}.${field.fieldName}`)}>
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
          </SortableContext>
        </DndContext>
      </ScrollableContent>
    </Box>
  );
};

export default memo(WorkflowLinearTab);
