import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Button, ButtonGroup, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { firefoxDndFix } from 'features/dnd/util';
import { FormElementComponent } from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import {
  buildAddFormElementDndData,
  useMonitorForFormElementDnd,
} from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import { formModeToggled, selectRootElementId, selectWorkflowFormMode } from 'features/nodes/store/workflowSlice';
import type { FormElement } from 'features/nodes/types/workflow';
import { buildContainer, buildDivider, buildHeading, buildText } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { assert } from 'tsafe';

export const WorkflowBuilder = memo(() => {
  const mode = useAppSelector(selectWorkflowFormMode);
  const rootElementId = useAppSelector(selectRootElementId);
  useMonitorForFormElementDnd();

  return (
    <ScrollableContent>
      <Flex justifyContent="center">
        <Flex flexDir="column" w={mode === 'view' ? '512px' : 'min-content'} minW="512px" gap={2}>
          <ButtonGroup isAttached={false} justifyContent="center">
            <ToggleModeButton />
            <AddFormElementDndButton type="row" />
            <AddFormElementDndButton type="column" />
            <AddFormElementDndButton type="divider" />
            <AddFormElementDndButton type="heading" />
            <AddFormElementDndButton type="text" />
          </ButtonGroup>
          {rootElementId && <FormElementComponent id={rootElementId} />}
          {!rootElementId && <EmptyState />}
        </Flex>
      </Flex>
    </ScrollableContent>
  );
});
WorkflowBuilder.displayName = 'WorkflowBuilder';

const EmptyState = memo(() => {
  const mode = useAppSelector(selectWorkflowFormMode);
  const dispatch = useAppDispatch();

  const toggleMode = useCallback(() => {
    dispatch(formModeToggled());
  }, [dispatch]);

  const addContainer = useCallback(() => {}, []);

  if (mode === 'view') {
    return (
      <Flex flexDir="column" gap={4} w="full" h="full" justifyContent="center" alignItems="center">
        <Text variant="subtext" fontSize="md">
          Click Edit to build a form for this workflow.
        </Text>
        <Button onClick={toggleMode}>Edit</Button>
      </Flex>
    );
  }

  <Flex flexDir="column" gap={4} w="full" h="full" justifyContent="center" alignItems="center">
    <Text variant="subtext" fontSize="md">
      No form elements added. Click a button above to add a form element.
    </Text>
  </Flex>;
});
EmptyState.displayName = 'EmptyState';

const ToggleModeButton = memo(() => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectWorkflowFormMode);

  const onClick = useCallback(() => {
    dispatch(formModeToggled());
  }, [dispatch]);

  return <Button onClick={onClick}>{mode === 'view' ? 'Edit' : 'View'}</Button>;
});
ToggleModeButton.displayName = 'ToggleModeButton';

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
            return buildAddFormElementDndData(element);
          }
          if (type === 'column') {
            const element = buildContainer('column', []);
            return buildAddFormElementDndData(element);
          }
          if (type === 'divider') {
            const element = buildDivider();
            return buildAddFormElementDndData(element);
          }
          if (type === 'heading') {
            const element = buildHeading('default heading');
            return buildAddFormElementDndData(element);
          }
          if (type === 'text') {
            const element = buildText('default text');
            return buildAddFormElementDndData(element);
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
  const rootElementId = useAppSelector(selectRootElementId);
  const isDragging = useAddFormElementDnd(type, draggableRef);

  return (
    <Button
      ref={draggableRef}
      variant="unstyled"
      borderWidth={2}
      borderStyle="dashed"
      borderRadius="base"
      px={4}
      py={1}
      cursor="grab"
      _hover={{ bg: 'base.800' }}
      isDisabled={isDragging || (type !== 'row' && type !== 'column' && !rootElementId)}
    >
      {startCase(type)}
    </Button>
  );
};
