import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { deepClone } from 'common/util/deepClone';
import { singleWorkflowFieldDndSource } from 'features/dnd/dnd';
import { triggerPostMoveFlash } from 'features/dnd/util';
import LinearViewFieldInternal from 'features/nodes/components/flow/nodes/Invocation/fields/LinearViewField';
import { selectWorkflowSlice, workflowExposedFieldsReordered } from 'features/nodes/store/workflowSlice';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { isEqual } from 'lodash-es';
import { memo, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => workflow.exposedFields);

const WorkflowLinearTab = () => {
  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} py={2} gap={2} h="full" w="full">
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
        if (!singleWorkflowFieldDndSource.typeGuard(source.data)) {
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
          !singleWorkflowFieldDndSource.typeGuard(sourceData) ||
          !singleWorkflowFieldDndSource.typeGuard(targetData)
        ) {
          return;
        }

        const fieldsClone = deepClone(fields);

        const indexOfSource = fieldsClone.findIndex((fieldIdentifier) =>
          isEqual(fieldIdentifier, sourceData.payload.fieldIdentifier)
        );
        const indexOfTarget = fieldsClone.findIndex((fieldIdentifier) =>
          isEqual(fieldIdentifier, targetData.payload.fieldIdentifier)
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
          list: fieldsClone,
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
        const element = document.querySelector(
          `[data-field-name="${sourceData.payload.fieldIdentifier.nodeId}-${sourceData.payload.fieldIdentifier.fieldName}"]`
        );
        if (element instanceof HTMLElement) {
          triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
        }
      },
    });
  }, [dispatch, fields]);

  return (
    <>
      {fields.map((fieldIdentifier) => (
        <LinearViewFieldInternal
          key={`${fieldIdentifier.nodeId}.${fieldIdentifier.fieldName}`}
          fieldIdentifier={fieldIdentifier}
        />
      ))}
    </>
  );
});

FieldListInnerContent.displayName = 'FieldListInnerContent';
