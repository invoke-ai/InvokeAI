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
import type { ContainerElement, ElementId, FormElement } from 'features/nodes/types/workflow';
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

type DndData = {
  element: FormElement;
  container: ContainerElement | null;
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
      // canMonitor({ source }) {
      //   return (source.data as FormElement).id === containerId;
      // },
      canMonitor: () => true,
      onDrop({ location, source }) {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceData = source.data as DndData;
        const targetData = target.data as DndData;

        const sourceElementId = sourceData.element.id;
        const targetElementId = targetData.element.id;

        const closestCenterOrEdge = extractClosestCenterOrEdge(targetData);

        if (closestCenterOrEdge === 'center') {
          const targetContainer = getElement(targetElementId);
          if (!isContainerElement(targetContainer)) {
            // Shouldn't happen - when dropped on the center of drop target, the target should always be a container type.
            return;
          }
          flushSync(() => {
            dispatch(formElementMoved({ id: sourceElementId, containerId: targetContainer.id }));
          });
        } else if (closestCenterOrEdge) {
          if (targetData.container) {
            const targetContainer = getElement(targetData.container.id);
            if (!isContainerElement(targetContainer)) {
              // Shouldn't happen - drop targets should always have a container.
              return;
            }
            const indexOfSource = targetContainer.data.children.findIndex((elementId) => elementId === sourceElementId);
            const indexOfTarget = targetContainer.data.children.findIndex((elementId) => elementId === targetElementId);

            if (indexOfSource === indexOfTarget) {
              // Don't move if the source and target are the same index, meaning same position in the list.
              return;
            }

            const adjustedIndex = adjustIndexForDrop(indexOfTarget, closestCenterOrEdge);

            if (indexOfSource === adjustedIndex) {
              // Don't move if the source is already in the correct position.
              return;
            }

            flushSync(() => {
              dispatch(
                formElementMoved({
                  id: sourceElementId,
                  containerId: targetContainer.id,
                  index: indexOfTarget,
                })
              );
            });
          }
        } else {
          // No container, cannot do anything
          return;
        }
        // const childrenClone = [...targetData.container.data.children];

        // const indexOfSource = childrenClone.findIndex((elementId) => elementId === sourceElementId);
        // const indexOfTarget = childrenClone.findIndex((elementId) => elementId === targetElementId);

        // if (indexOfTarget < 0 || indexOfSource < 0) {
        //   return;
        // }

        // // Don't move if the source and target are the same index, meaning same position in the list
        // if (indexOfSource === indexOfTarget) {
        //   return;
        // }

        // Using `flushSync` so we can query the DOM straight after this line
        // flushSync(() => {
        //   dispatch(
        //     formElementMoved({
        //       id: sourceElementId,
        //       containerId: targetData.container.id,
        //       index: indexOfTarget,
        //     })
        //   );
        // });

        // Flash the element that was moved
        const element = document.querySelector(`#${getEditModeWrapperId(sourceElementId)}`);
        if (element instanceof HTMLElement) {
          triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
        }
      },
    });
  }, [dispatch]);
};

export const useDraggableFormElement = (
  elementId: ElementId,
  containerId: ElementId | null,
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
        canDrag: () => Boolean(containerId),
        element: draggableElement,
        dragHandle: dragHandleElement,
        getInitialData() {
          const data: DndData = {
            element: getElement(elementId),
            container: containerId ? getElement(containerId, isContainerElement) : null,
          };
          return data;
        },
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
        // canDrop() {},
        getData({ input }) {
          const element = getElement(elementId);
          const container = containerId ? getElement(containerId, isContainerElement) : null;

          const data: DndData = {
            element,
            container,
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
        getIsSticky() {
          return true;
        },
        onDrag({ self, location }) {
          const innermostDropTargetElement = location.current.dropTargets.at(0)?.element;

          // If the innermost target is not this draggable element, bail. We only want to react when dragging over _this_ element.
          if (!innermostDropTargetElement || innermostDropTargetElement !== draggableElement) {
            setListDndState(idle);
            return;
          }

          const closestCenterOrEdge = extractClosestCenterOrEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setListDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestCenterOrEdge === closestCenterOrEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestCenterOrEdge };
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
  }, [containerId, dragHandleRef, draggableRef, elementId]);

  return [dndListState, isDragging] as const;
};
