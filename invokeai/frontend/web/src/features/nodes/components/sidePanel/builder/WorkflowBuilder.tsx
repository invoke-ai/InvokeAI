import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
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
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { assert } from 'tsafe';

export const WorkflowBuilder = memo(() => {
  const mode = useAppSelector(selectWorkflowFormMode);
  const rootElementId = useAppSelector(selectRootElementId);
  useMonitorForFormElementDnd();

  return (
    <ScrollableContent>
      <Flex w="full" justifyContent="center">
        <Flex flexDir="column" w={mode === 'view' ? '512px' : 'min-content'} minW="512px" gap={2}>
          <ButtonGroup isAttached={false} justifyContent="center">
            <ToggleModeButton />
            <AddFormElementDndButton type="container" />
            <AddFormElementDndButton type="divider" />
            <AddFormElementDndButton type="heading" />
            <AddFormElementDndButton type="text" />
          </ButtonGroup>
          {rootElementId && <FormElementComponent id={rootElementId} />}
        </Flex>
      </Flex>
    </ScrollableContent>
  );
});

WorkflowBuilder.displayName = 'WorkflowBuilder';

const ToggleModeButton = memo(() => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectWorkflowFormMode);

  const onClick = useCallback(() => {
    dispatch(formModeToggled());
  }, [dispatch]);

  return <Button onClick={onClick}>{mode === 'view' ? 'Edit' : 'View'}</Button>;
});
ToggleModeButton.displayName = 'ToggleModeButton';

const useAddFormElementDnd = (type: Omit<FormElement['type'], 'node-field'>, draggableRef: RefObject<HTMLElement>) => {
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
        getInitialData: () => {
          if (type === 'container') {
            const element = buildContainer('row', []);
            return buildAddFormElementDndData(element);
          }
          if (type === 'divider') {
            const element = buildDivider();
            return buildAddFormElementDndData(element);
          }
          if (type === 'heading') {
            const element = buildHeading('default heading', 1);
            return buildAddFormElementDndData(element);
          }
          if (type === 'text') {
            const element = buildText('default text', 'sm');
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
  }, [draggableRef, type]);

  return isDragging;
};

const AddFormElementDndButton = ({ type }: { type: Omit<FormElement['type'], 'node-field'> }) => {
  const draggableRef = useRef<HTMLButtonElement>(null);
  const isDragging = useAddFormElementDnd(type, draggableRef);

  return (
    <Button ref={draggableRef} variant="ghost" pointerEvents="auto" opacity={isDragging ? 0.3 : 1}>
      {type}
    </Button>
  );
};
