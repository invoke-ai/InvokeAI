import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { Dnd, triggerPostMoveFlash } from 'features/dnd/dnd';
import LinearViewFieldInternal from 'features/nodes/components/flow/nodes/Invocation/fields/LinearViewField';
import { selectWorkflowSlice, workflowExposedFieldsReordered } from 'features/nodes/store/workflowSlice';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { memo, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => workflow.exposedFields);

const WorkflowLinearTab = () => {
  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} h="full" w="full">
          <FieldListContent />
        </Flex>
      </ScrollableContent>
    </Box>
  );
};

export default memo(WorkflowLinearTab);

const FieldListContent = memo(() => {
  const fields = useAppSelector(selector);
  const { isLoading } = useGetOpenAPISchemaQuery();
  const { t } = useTranslation();

  if (isLoading) {
    return <IAINoContentFallback label={t('nodes.loadingNodes')} icon={null} />;
  }

  if (fields.length === 0) {
    <IAINoContentFallback label={t('nodes.noFieldsLinearview')} icon={null} />;
  }

  return <FieldListInnerContent fields={fields} />;
});

FieldListContent.displayName = 'FieldListContent';

const FieldListInnerContent = memo(({ fields }: { fields: FieldIdentifier[] }) => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    return monitorForElements({
      canMonitor({ source }) {
        if (!Dnd.Source.singleWorkflowField.typeGuard(source.data)) {
          return false;
        }
        return true;
      },
      onDrop({ location, source }) {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceData = source.data;
        const targetData = target.data;

        if (
          !Dnd.Source.singleWorkflowField.typeGuard(sourceData) ||
          !Dnd.Source.singleWorkflowField.typeGuard(targetData)
        ) {
          return;
        }

        const indexOfSource = fields.findIndex(
          (fieldIdentifier) => fieldIdentifier.fieldName === sourceData.payload.fieldIdentifier.fieldName
        );
        const indexOfTarget = fields.findIndex(
          (fieldIdentifier) => fieldIdentifier.fieldName === targetData.payload.fieldIdentifier.fieldName
        );

        if (indexOfTarget < 0 || indexOfSource < 0) {
          return;
        }

        // Don't move if the source and target are the same index, meaning same position in the list
        if (indexOfSource === indexOfTarget) {
          return;
        }

        const closestEdgeOfTarget = extractClosestEdge(targetData);

        // It's possible that the indices are different, but refer to the same position. For example, if the source is
        // at 2 and the target is at 3, but the target edge is 'top', then the entity is already in the correct position.
        // We should bail if this is the case.
        let edgeIndexDelta = 0;

        if (closestEdgeOfTarget === 'bottom') {
          edgeIndexDelta = 1;
        } else if (closestEdgeOfTarget === 'top') {
          edgeIndexDelta = -1;
        }

        // If the source is already in the correct position, we don't need to move it.
        if (indexOfSource === indexOfTarget + edgeIndexDelta) {
          return;
        }

        const reorderedFields = reorderWithEdge({
          list: fields,
          startIndex: indexOfSource,
          indexOfTarget,
          closestEdgeOfTarget,
          axis: 'vertical',
        });

        // Using `flushSync` so we can query the DOM straight after this line
        flushSync(() => {
          dispatch(workflowExposedFieldsReordered(reorderedFields));
        });

        // Flash the element that was moved
        const element = document.querySelector(`[data-field-name="${sourceData.payload.fieldIdentifier.fieldName}"]`);
        if (element instanceof HTMLElement) {
          triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
        }
      },
    });
  }, [dispatch, fields]);

  return (
    <>
      {fields.map(({ nodeId, fieldName }) => (
        <LinearViewFieldInternal key={`${nodeId}.${fieldName}`} nodeId={nodeId} fieldName={fieldName} />
      ))}
    </>
  );
});

FieldListInnerContent.displayName = 'FieldListInnerContent';
