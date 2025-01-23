import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import {
  draggable,
  dropTargetForElements,
  monitorForElements,
} from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import type { DndListTargetState } from 'features/dnd/types';
import { idle } from 'features/dnd/types';
import { firefoxDndFix, triggerPostMoveFlash } from 'features/dnd/util';
import { formElementContainerDataChanged } from 'features/nodes/store/workflowSlice';
import type { ElementId, FormElement } from 'features/nodes/types/workflow';
import { isContainerElement } from 'features/nodes/types/workflow';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { assert } from 'tsafe';

export const useMonitorForFormElementDnd = (containerId: string, children: ElementId[]) => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    return monitorForElements({
      canMonitor({ source }) {
        return (source.data as FormElement).id === containerId;
      },
      onDrop({ location, source }) {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceData = source.data as FormElement;
        const targetData = target.data as FormElement;

        const sourceElementId = sourceData.id;
        const targetElementId = targetData.id;

        const childrenClone = [...children];

        const indexOfSource = childrenClone.findIndex((elementId) => elementId === sourceElementId);
        const indexOfTarget = childrenClone.findIndex((elementId) => elementId === targetElementId);

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
        // let edgeIndexDelta = 0;

        // if (closestEdgeOfTarget === 'bottom') {
        //   edgeIndexDelta = 1;
        // } else if (closestEdgeOfTarget === 'top') {
        //   edgeIndexDelta = -1;
        // }

        // If the source is already in the correct position, we don't need to move it.
        // if (indexOfSource === indexOfTarget + edgeIndexDelta) {
        //   return;
        // }

        const reorderedChildren = reorderWithEdge({
          list: childrenClone,
          startIndex: indexOfSource,
          indexOfTarget,
          closestEdgeOfTarget,
          axis: 'vertical',
        });

        // Using `flushSync` so we can query the DOM straight after this line
        flushSync(() => {
          dispatch(formElementContainerDataChanged({ id: containerId, changes: { children: reorderedChildren } }));
        });

        // Flash the element that was moved
        const element = document.querySelector(`#${sourceElementId}`);
        if (element instanceof HTMLElement) {
          triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
        }
      },
    });
  }, [children, containerId, dispatch]);
};

const getElement = (id: ElementId) => {
  const el = getStore().getState().workflow.form?.elements[id];
  assert(el !== undefined);
  return el;
};

export const useDraggableFormElement = (
  elementId: ElementId,
  draggableRef: RefObject<HTMLElement>,
  dragHandleRef: RefObject<HTMLElement>
) => {
  const [dndListState, setListDndState] = useState<DndListTargetState>(idle);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const draggableElement = draggableRef.current;
    const dragHandleElement = dragHandleRef.current;
    if (!draggableElement || !dragHandleElement) {
      return;
    }
    return combine(
      firefoxDndFix(draggableElement),
      draggable({
        element: draggableElement,
        dragHandle: dragHandleElement,
        getInitialData() {
          return getElement(elementId);
        },
        // getInitialData() {
        //   return singleWorkflowFieldDndSource.getData({ fieldIdentifier });
        // },
        onDragStart() {
          setListDndState({ type: 'is-dragging' });
          setIsDragging(true);
        },
        onDrop() {
          setListDndState(idle);
          setIsDragging(false);
        },
      }),
      dropTargetForElements({
        element: draggableElement,
        canDrop() {
          return isContainerElement(getElement(elementId));
        },
        getData({ input }) {
          const data = { elementId };
          return attachClosestEdge(data, {
            element: draggableElement,
            input,
            allowedEdges: ['top', 'bottom', 'left', 'right'],
          });
        },
        getIsSticky() {
          return true;
        },
        onDragEnter({ self }) {
          const closestEdge = extractClosestEdge(self.data);
          setListDndState({ type: 'is-dragging-over', closestEdge });
          console.log('onDragEnter', self.data);
        },
        onDrag({ self }) {
          const closestEdge = extractClosestEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setListDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestEdge === closestEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestEdge };
          });
        },
        onDragLeave() {
          setListDndState(idle);
        },
        onDrop() {
          setListDndState(idle);
        },
      })
    );
  }, [dragHandleRef, draggableRef, elementId]);

  return [dndListState, isDragging] as const;
};
