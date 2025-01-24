import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import {
  draggable,
  dropTargetForElements,
  monitorForElements,
} from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { firefoxDndFix, triggerPostMoveFlash } from 'features/dnd/util';
import type { CenterOrEdge } from 'features/nodes/components/sidePanel/builder/center-or-closest-edge';
import {
  attachClosestCenterOrEdge,
  extractClosestCenterOrEdge,
} from 'features/nodes/components/sidePanel/builder/center-or-closest-edge';
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { formElementMoved } from 'features/nodes/store/workflowSlice';
import type { ElementId, FormElement } from 'features/nodes/types/workflow';
import { isContainerElement } from 'features/nodes/types/workflow';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { assert } from 'tsafe';

/**
 * States for a dnd list with containers.
 */
export type DndListTargetState =
  | {
      type: 'idle';
    }
  | {
      type: 'preview';
      container: HTMLElement;
    }
  | {
      type: 'is-dragging';
    }
  | {
      type: 'is-dragging-over';
      closestCenterOrEdge: CenterOrEdge | null;
    };
export const idle: DndListTargetState = { type: 'idle' };

// using a symbol so we can guarantee a key with a unique value
const uniqueBuilderDndKey = Symbol('builderDnd');

type DndData = {
  [uniqueBuilderDndKey]: true;
  element: FormElement;
};

const getElement = <T extends FormElement>(id: ElementId, guard?: (el: FormElement) => el is T): T => {
  const el = getStore().getState().workflow.form?.elements[id];
  assert(el);
  if (guard) {
    assert(guard(el));
    return el;
  } else {
    return el as T;
  }
};

const adjustIndexForDrop = (index: number, edge: Exclude<CenterOrEdge, 'center'>) => {
  if (edge === 'left' || edge === 'top') {
    return index - 1;
  }
  return index + 1;
};

export const useMonitorForFormElementDnd = () => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    return monitorForElements({
      canMonitor: ({ source }) => uniqueBuilderDndKey in source.data,
      onDrop: ({ location, source }) => {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceData = source.data as DndData;
        const targetData = target.data as DndData;

        //
        if (sourceData.element.id === targetData.element.id) {
          return;
        }

        const closestCenterOrEdge = extractClosestCenterOrEdge(targetData);

        if (closestCenterOrEdge === 'center') {
          // Move the element to the target container - should we double-check that the target is a container?
          flushSync(() => {
            dispatch(formElementMoved({ id: sourceData.element.id, containerId: targetData.element.id }));
          });
        } else if (closestCenterOrEdge) {
          // Move the element to the target's parent container at the correct index
          const { parentId } = targetData.element;
          assert(parentId !== undefined, 'Target element should have a parent');

          const isReparenting = parentId !== sourceData.element.parentId;

          const parentContainer = getElement(parentId, isContainerElement);
          const targetIndex = parentContainer.data.children.findIndex(
            (elementId) => elementId === targetData.element.id
          );

          let index: number | undefined = undefined;

          if (!isReparenting) {
            const sourceIndex = parentContainer.data.children.findIndex(
              (elementId) => elementId === sourceData.element.id
            );
            if (sourceIndex === targetIndex || sourceIndex === adjustIndexForDrop(targetIndex, closestCenterOrEdge)) {
              return;
            }
            index = targetIndex;
          } else {
            index = adjustIndexForDrop(targetIndex, closestCenterOrEdge);
          }

          flushSync(() => {
            dispatch(
              formElementMoved({
                id: sourceData.element.id,
                containerId: parentId,
                index,
              })
            );
          });
        } else {
          // No container, cannot do anything
          return;
        }

        // Flash the element that was moved
        const element = document.querySelector(`#${getEditModeWrapperId(sourceData.element.id)}`);
        if (element instanceof HTMLElement) {
          triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
        }
      },
    });
  }, [dispatch]);
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
    const _element = getElement(elementId);
    if (!_element.parentId) {
      // Root element, cannot drag
      return;
    }
    return combine(
      firefoxDndFix(draggableElement),
      draggable({
        element: draggableElement,
        dragHandle: dragHandleElement,
        getInitialData: () => ({
          [uniqueBuilderDndKey]: true,
          element: getElement(elementId),
        }),
        onDragStart: () => {
          setListDndState({ type: 'is-dragging' });
          setIsDragging(true);
        },
        onDrop: () => {
          setListDndState(idle);
          setIsDragging(false);
        },
      }),
      dropTargetForElements({
        element: draggableElement,
        canDrop: ({ source }) => uniqueBuilderDndKey in source.data,
        getData: ({ input }) => {
          const element = getElement(elementId);
          const container = element.parentId ? getElement(element.parentId, isContainerElement) : null;

          const data: DndData = {
            [uniqueBuilderDndKey]: true,
            element,
          };

          const allowedCenterOrEdge: CenterOrEdge[] = [];

          if (isContainerElement(element)) {
            allowedCenterOrEdge.push('center');
          }

          if (container?.data.direction === 'row') {
            allowedCenterOrEdge.push('left', 'right');
          }

          if (container?.data.direction === 'column') {
            allowedCenterOrEdge.push('top', 'bottom');
          }

          return attachClosestCenterOrEdge(data, {
            element: draggableElement,
            input,
            allowedCenterOrEdge,
          });
        },
        getIsSticky: () => true,
        onDrag: ({ self, location, source }) => {
          const innermostDropTargetElement = location.current.dropTargets.at(0)?.element;

          // If the innermost target is not this draggable element, bail. We only want to react when dragging over _this_ element.
          if (!innermostDropTargetElement || innermostDropTargetElement !== draggableElement) {
            setListDndState(idle);
            return;
          }

          const closestCenterOrEdge = extractClosestCenterOrEdge(self.data);

          // Don't allow reparanting to the same container
          if (closestCenterOrEdge === 'center' && source.element === draggableElement) {
            setListDndState(idle);
            return;
          }

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setListDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestCenterOrEdge === closestCenterOrEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestCenterOrEdge };
          });
        },
        onDragLeave: () => {
          setListDndState(idle);
        },
        onDrop: () => {
          setListDndState(idle);
        },
      })
    );
  }, [dragHandleRef, draggableRef, elementId]);

  return [dndListState, isDragging] as const;
};
