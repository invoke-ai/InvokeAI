import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Button, ButtonGroup, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { firefoxDndFix } from 'features/dnd/util';
import { FormElementComponent } from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import {
  buildFormElementDndData,
  useBuilderDndMonitor,
  useRootDnd,
} from 'features/nodes/components/sidePanel/builder/dnd';
import {
  formModeChanged,
  formReset,
  selectFormIsEmpty,
  selectFormLayout,
  selectWorkflowFormMode,
} from 'features/nodes/store/workflowSlice';
import type { FormElement } from 'features/nodes/types/workflow';
import { buildContainer, buildDivider, buildHeading, buildText } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

export const WorkflowBuilder = memo(() => {
  const { t } = useTranslation();
  const mode = useAppSelector(selectWorkflowFormMode);
  const dispatch = useAppDispatch();
  const isEmpty = useAppSelector(selectFormIsEmpty);
  useBuilderDndMonitor();

  const resetForm = useCallback(() => {
    dispatch(formReset());
  }, [dispatch]);

  const setToViewMode = useCallback(() => {
    dispatch(formModeChanged({ mode: 'view' }));
  }, [dispatch]);

  const setToEditMode = useCallback(() => {
    dispatch(formModeChanged({ mode: 'edit' }));
  }, [dispatch]);

  return (
    <ScrollableContent>
      <Flex justifyContent="center" w="full" h="full" p={4}>
        <Flex flexDir="column" w={mode === 'view' ? '512px' : 'min-content'} h="full" minW="512px" gap={4}>
          {mode === 'edit' && (
            <ButtonGroup isAttached={false} justifyContent="center">
              <AddFormElementDndButton type="row" />
              <AddFormElementDndButton type="column" />
              <AddFormElementDndButton type="divider" />
              <AddFormElementDndButton type="heading" />
              <AddFormElementDndButton type="text" />
              <Button onClick={setToViewMode}>{t('common.view')}</Button>
              <Button onClick={resetForm}>{t('common.reset')}</Button>
            </ButtonGroup>
          )}
          {mode === 'view' && !isEmpty && <Button onClick={setToEditMode}>{t('common.edit')}</Button>}
          {!isEmpty && <FormLayout />}
          {mode === 'view' && isEmpty && <EmptyStateViewMode />}
          {mode === 'edit' && isEmpty && <EmptyStateEditMode />}
        </Flex>
      </Flex>
    </ScrollableContent>
  );
});
WorkflowBuilder.displayName = 'WorkflowBuilder';

const FormLayout = memo(() => {
  const layout = useAppSelector(selectFormLayout);

  return (
    <Flex flexDir="column" gap={4} w="full" p={4} borderRadius="base">
      {layout.map((id) => (
        <FormElementComponent key={id} id={id} />
      ))}
    </Flex>
  );
});
FormLayout.displayName = 'FormLayout';

const EmptyStateViewMode = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const setToEditMode = useCallback(() => {
    dispatch(formModeChanged({ mode: 'edit' }));
  }, [dispatch]);

  return (
    <Flex flexDir="column" gap={4} w="full" h="full" justifyContent="center" alignItems="center">
      <Text variant="subtext" fontSize="md">
        {t('workflows.builder.emptyRootPlaceholderViewMode')}
      </Text>
      <Button onClick={setToEditMode}>{t('common.edit')}</Button>
    </Flex>
  );
});
EmptyStateViewMode.displayName = 'EmptyStateViewMode';

const EmptyStateEditMode = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const isDragging = useRootDnd(ref);

  return (
    <Flex
      id={getEditModeWrapperId('root')}
      ref={ref}
      w="full"
      h="full"
      bg={isDragging ? 'base.800' : undefined}
      p={4}
      borderRadius="base"
      alignItems="center"
      justifyContent="center"
    >
      <Text variant="subtext" fontSize="md">
        {t('workflows.builder.emptyRootPlaceholderEditMode')}
      </Text>
    </Flex>
  );
});
EmptyStateEditMode.displayName = 'EmptyStateEditMode';

const useAddFormElementDnd = (
  type: Exclude<FormElement['type'], 'node-field' | 'container'> | 'row' | 'column',
  draggableRef: RefObject<HTMLElement>,
  isEnabled = true
) => {
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const draggableElement = draggableRef.current;
    if (!draggableElement) {
      return;
    }
    return combine(
      firefoxDndFix(draggableElement),
      draggable({
        element: draggableElement,
        canDrag: () => isEnabled,
        getInitialData: () => {
          if (type === 'row') {
            const element = buildContainer('row', []);
            return buildFormElementDndData(element);
          }
          if (type === 'column') {
            const element = buildContainer('column', []);
            return buildFormElementDndData(element);
          }
          if (type === 'divider') {
            const element = buildDivider();
            return buildFormElementDndData(element);
          }
          if (type === 'heading') {
            const element = buildHeading('default heading');
            return buildFormElementDndData(element);
          }
          if (type === 'text') {
            const element = buildText('default text');
            return buildFormElementDndData(element);
          }
          assert(false);
        },
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
      })
    );
  }, [draggableRef, isEnabled, type]);

  return isDragging;
};

const AddFormElementDndButton = ({ type }: { type: Parameters<typeof useAddFormElementDnd>[0] }) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const isDragging = useAddFormElementDnd(type, draggableRef);

  return (
    <Button
      as="div"
      ref={draggableRef}
      pointerEvents="all"
      variant="unstyled"
      borderWidth={2}
      borderStyle="dashed"
      borderRadius="base"
      px={4}
      py={1}
      cursor="grab"
      _hover={{ bg: 'base.800' }}
      isDisabled={isDragging}
    >
      {startCase(type)}
    </Button>
  );
};
