import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import {
  draggable,
  dropTargetForElements,
  monitorForElements,
} from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { getReorderDestinationIndex } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/get-reorder-destination-index';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { firefoxDndFix, triggerPostMoveFlash } from 'features/dnd/util';
import type { CenterOrEdge } from 'features/nodes/components/sidePanel/builder/center-or-closest-edge';
import {
  attachClosestCenterOrEdge,
  extractClosestCenterOrEdge,
} from 'features/nodes/components/sidePanel/builder/center-or-closest-edge';
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import {
  formContainerChildrenReordered,
  formElementAdded,
  formElementReparented,
  formRootReordered,
  selectFormIsEmpty,
} from 'features/nodes/store/workflowSlice';
import type { FieldIdentifier, FieldInputTemplate } from 'features/nodes/types/field';
import type { ElementId, FormElement } from 'features/nodes/types/workflow';
import { buildNodeFieldElement, isContainerElement } from 'features/nodes/types/workflow';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { assert } from 'tsafe';

const log = logger('dnd');

const uniqueRootContainerKey = Symbol('root-container');
type RootContainerDndData = {
  [uniqueRootContainerKey]: true;
};
const buildRootContainerDndData = (): RootContainerDndData => ({
  [uniqueRootContainerKey]: true,
});
const isRootContainerDndData = (data: Record<string | symbol, unknown>): data is RootContainerDndData => {
  return uniqueRootContainerKey in data;
};

const uniqueFormElementDndKey = Symbol('form-element');
type FormElementDndData = {
  [uniqueFormElementDndKey]: true;
  element: FormElement;
};
export const buildFormElementDndData = (element: FormElement): FormElementDndData => ({
  [uniqueFormElementDndKey]: true,
  element,
});
const isFormElementDndData = (data: Record<string | symbol, unknown>): data is FormElementDndData => {
  return uniqueFormElementDndKey in data;
};

const elementExists = (id: ElementId): boolean => {
  return getStore().getState().workflow.form?.elements[id] !== undefined;
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

const getLayout = () => {
  return getStore().getState().workflow.form.layout;
};

const flashElement = (elementId: ElementId) => {
  const element = document.querySelector(`#${getEditModeWrapperId(elementId)}`);
  if (element instanceof HTMLElement) {
    triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
  }
};

export const useMonitorForFormElementDnd = () => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    return monitorForElements({
      canMonitor: ({ source }) => isFormElementDndData(source.data),
      onDrop: ({ location, source }) => {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const targetData = target.data;
        const sourceData = source.data;

        if (!isFormElementDndData(targetData) && !isRootContainerDndData(targetData)) {
          return;
        }

        if (!isFormElementDndData(sourceData)) {
          return;
        }

        const isAddingNewElement = !elementExists(sourceData.element.id);

        // Appending a new element to the root container
        if (isAddingNewElement && isRootContainerDndData(targetData)) {
          log.debug('Adding new element to empty root');
          flushSync(() => {
            dispatch(
              formElementAdded({
                element: sourceData.element,
                containerId: undefined,
                index: undefined,
              })
            );
          });
          flashElement(sourceData.element.id);
          return;
        }

        // Inserting a new element - maybe to root, maybe to a container
        if (isAddingNewElement && isFormElementDndData(targetData) && targetData.element.parentId === undefined) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          // Adding to an empty container
          if (closestEdgeOfTarget === 'center') {
            log.debug('Adding new element to empty container');
            assert(isContainerElement(targetData.element), 'Expected target to be a container');
            flushSync(() => {
              dispatch(
                formElementAdded({
                  element: sourceData.element,
                  containerId: targetData.element.id,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          } else {
            // Adding to the root
            log.debug('Inserting new element into root');
            // closestEdgeOfTarget !== 'center'
            const layout = getLayout();
            const indexOfTarget = layout.indexOf(targetData.element.id);
            const index = getReorderDestinationIndex({
              startIndex: indexOfTarget + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: 'vertical',
            });
            flushSync(() => {
              dispatch(
                formElementAdded({
                  element: sourceData.element,
                  containerId: undefined,
                  index,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          }
        }

        // Inserting a new element into a container
        if (isAddingNewElement && isFormElementDndData(targetData) && targetData.element.parentId !== undefined) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          // Adding to an empty container
          if (closestEdgeOfTarget === 'center') {
            log.debug('Adding new element to empty container');
            assert(isContainerElement(targetData.element), 'Expected target to be a container');
            flushSync(() => {
              dispatch(
                formElementAdded({
                  element: sourceData.element,
                  containerId: targetData.element.id,
                  index: undefined,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          } else {
            log.debug('Inserting new element into container');
            const container = getElement(targetData.element.parentId, isContainerElement);
            const indexOfTarget = container.data.children.indexOf(targetData.element.id);
            const index = getReorderDestinationIndex({
              startIndex: indexOfTarget + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            flushSync(() => {
              dispatch(
                formElementAdded({
                  element: sourceData.element,
                  containerId: container.id,
                  index,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          }
        }

        // Reparenting an existing element to the root, appending it to the end
        if (!isAddingNewElement && isRootContainerDndData(targetData) && sourceData.element.parentId !== undefined) {
          log.debug('Reparenting element from container to empty root');

          flushSync(() => {
            dispatch(
              formElementReparented({
                id: sourceData.element.id,
                containerId: undefined,
                index: undefined,
              })
            );
          });
          flashElement(sourceData.element.id);
          return;
        }

        // This should never happen - the root is a dnd target _only_ when it is empty and we are adding a new element!
        if (!isAddingNewElement && isRootContainerDndData(targetData)) {
          log.error('Attempted to move an existing element to the root directly!');
          return;
        }

        // Moving an existing element within the root
        if (
          !isAddingNewElement &&
          isFormElementDndData(targetData) &&
          targetData.element.parentId === undefined &&
          sourceData.element.parentId === undefined
        ) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          if (closestEdgeOfTarget === 'center') {
            log.debug('Reparenting element from root to empty container');
            assert(isContainerElement(targetData.element), 'Expected target to be a container');
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: targetData.element.id,
                  index: undefined,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          } else {
            log.debug('Moving element within root');
            const layout = getLayout();
            const startIndex = layout.indexOf(sourceData.element.id);
            const indexOfTarget = layout.indexOf(targetData.element.id);
            const reorderedLayout = reorderWithEdge({
              list: layout,
              startIndex,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: 'vertical',
            });
            flushSync(() => {
              dispatch(
                formRootReordered({
                  layout: reorderedLayout,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          }
        }

        // Moving an existing element within a container
        if (
          !isAddingNewElement &&
          isFormElementDndData(targetData) &&
          targetData.element.parentId !== undefined &&
          sourceData.element.parentId !== undefined &&
          targetData.element.parentId === sourceData.element.parentId
        ) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          if (closestEdgeOfTarget === 'center') {
            log.debug('Reparenting element from a container to an empty container with same parent');
            log.debug('Reparenting element from one container to an empty container');
            assert(isContainerElement(targetData.element), 'Expected target to be a container');
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: targetData.element.id,
                  index: undefined,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          } else {
            log.debug('Moving element within container');
            const container = getElement(targetData.element.parentId, isContainerElement);
            const startIndex = container.data.children.indexOf(sourceData.element.id);
            const indexOfTarget = container.data.children.indexOf(targetData.element.id);
            const reorderedLayout = reorderWithEdge({
              list: container.data.children,
              startIndex,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            flushSync(() => {
              dispatch(
                formContainerChildrenReordered({
                  containerId: container.id,
                  children: reorderedLayout,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          }
        }

        // Moving an existing element within a container
        if (
          !isAddingNewElement &&
          isFormElementDndData(targetData) &&
          targetData.element.parentId !== undefined &&
          sourceData.element.parentId !== undefined &&
          targetData.element.parentId !== sourceData.element.parentId
        ) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          if (closestEdgeOfTarget === 'center') {
            log.debug('Reparenting element from one container to an empty container with different parent');
            assert(isContainerElement(targetData.element), 'Expected target to be a container');
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: targetData.element.id,
                  index: undefined,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          } else {
            log.debug('Moving element from one container to another');
            const container = getElement(targetData.element.parentId, isContainerElement);
            const indexOfTarget = container.data.children.indexOf(targetData.element.id);
            const index = getReorderDestinationIndex({
              startIndex: container.data.children.length + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: container.id,
                  index,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          }
        }

        // Moving an existing element from a container to root
        if (
          !isAddingNewElement &&
          isFormElementDndData(targetData) &&
          targetData.element.parentId === undefined &&
          sourceData.element.parentId !== undefined
        ) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          if (closestEdgeOfTarget === 'center') {
            log.debug('Reparenting element from container to empty container');
            assert(isContainerElement(targetData.element), 'Expected target to be a container');
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: targetData.element.id,
                  index: undefined,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          } else {
            log.debug('Reparenting element from container to root');
            const layout = getLayout();
            const indexOfTarget = layout.indexOf(targetData.element.id);
            const index = getReorderDestinationIndex({
              startIndex: layout.length + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: 'vertical',
            });
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: undefined,
                  index,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          }
        }

        // Moving an existing element from root to a container
        if (
          !isAddingNewElement &&
          isFormElementDndData(targetData) &&
          targetData.element.parentId !== undefined &&
          sourceData.element.parentId === undefined
        ) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          if (closestEdgeOfTarget === 'center') {
            log.debug('Reparenting element from root to empty container');
            assert(isContainerElement(targetData.element), 'Expected target to be a container');
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: targetData.element.id,
                  index: undefined,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          } else {
            log.debug('Reparenting element from root to container');
            const container = getElement(targetData.element.parentId, isContainerElement);
            const indexOfTarget = container.data.children.indexOf(targetData.element.id);
            const index = getReorderDestinationIndex({
              startIndex: indexOfTarget + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            flushSync(() => {
              dispatch(
                formElementReparented({
                  id: sourceData.element.id,
                  containerId: targetData.element.parentId,
                  index,
                })
              );
            });
            flashElement(sourceData.element.id);
            return;
          }
        }

        log.warn({ targetData, sourceData }, 'Unhandled drop event!');
      },
    });
  }, [dispatch]);
};

export const useRootContainerDropTarget = (ref: RefObject<HTMLElement>) => {
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const isEmpty = useAppSelector(selectFormIsEmpty);
  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return dropTargetForElements({
      element,
      canDrop: ({ source }) => isFormElementDndData(source.data) && isEmpty,
      getData: () => buildRootContainerDndData(),
      onDrag: ({ location, source }) => {
        const innermostDropTargetElement = location.current.dropTargets.at(0)?.element;

        // If the innermost target is not this draggable element, bail. We only want to react when dragging over _this_ element.
        if (!innermostDropTargetElement || innermostDropTargetElement !== element) {
          setIsDraggingOver(false);
          return;
        }

        // Don't allow reparanting to the same container
        if (source.element === element) {
          setIsDraggingOver(false);
          return;
        }
        setIsDraggingOver(true);
      },
      onDragLeave: () => {
        setIsDraggingOver(false);
      },
      onDrop: () => {
        setIsDraggingOver(false);
      },
    });
  }, [isEmpty, ref]);

  return isDraggingOver;
};

export const useDraggableFormElement = (
  elementId: ElementId,
  draggableRef: RefObject<HTMLElement>,
  dragHandleRef: RefObject<HTMLElement>
) => {
  const [isDragging, setIsDragging] = useState(false);
  const [activeDropRegion, setActiveDropRegion] = useState<CenterOrEdge | null>(null);

  useEffect(() => {
    const draggableElement = draggableRef.current;
    const dragHandleElement = dragHandleRef.current;
    if (!draggableElement || !dragHandleElement) {
      return;
    }
    const _element = getElement(elementId);

    return combine(
      firefoxDndFix(draggableElement),
      draggable({
        element: draggableElement,
        dragHandle: dragHandleElement,
        getInitialData: () => buildFormElementDndData(getElement(elementId)),
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
      }),
      dropTargetForElements({
        element: draggableElement,
        canDrop: ({ source }) =>
          isFormElementDndData(source.data) && source.data.element.id !== getElement(elementId).parentId,
        getData: ({ input }) => {
          const element = getElement(elementId);
          const targetData = buildFormElementDndData(element);

          const allowedCenterOrEdge: CenterOrEdge[] = [];

          if (isContainerElement(element) && element.data.children.length === 0) {
            allowedCenterOrEdge.push('center');
          }

          if (element.parentId !== undefined) {
            const parentContainer = getElement(element.parentId, isContainerElement);
            if (parentContainer.data.layout === 'row') {
              allowedCenterOrEdge.push('left', 'right');
            } else {
              // parentContainer.data.layout === 'row'
              allowedCenterOrEdge.push('top', 'bottom');
            }
          }

          if (element.parentId === undefined) {
            // Root container
            allowedCenterOrEdge.push('top', 'bottom');
          }

          return attachClosestCenterOrEdge(targetData, {
            element: draggableElement,
            input,
            allowedCenterOrEdge,
          });
        },
        onDrag: ({ self, location, source }) => {
          const innermostDropTargetElement = location.current.dropTargets.at(0)?.element;

          // If the innermost target is not this draggable element, bail. We only want to react when dragging over _this_ element.
          if (!innermostDropTargetElement || innermostDropTargetElement !== draggableElement) {
            setActiveDropRegion(null);
            return;
          }

          const closestCenterOrEdge = extractClosestCenterOrEdge(self.data);

          // Don't allow reparanting to the same container
          if (closestCenterOrEdge === 'center' && source.element === draggableElement) {
            setActiveDropRegion(null);
            return;
          }

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setActiveDropRegion(closestCenterOrEdge);
        },
        onDragLeave: () => {
          setActiveDropRegion(null);
        },
        onDrop: () => {
          setActiveDropRegion(null);
        },
      })
    );
  }, [dragHandleRef, draggableRef, elementId]);

  return [activeDropRegion, isDragging] as const;
};

export const useNodeFieldDnd = (
  fieldIdentifier: FieldIdentifier,
  fieldTemplate: FieldInputTemplate,
  draggableRef: RefObject<HTMLElement>,
  dragHandleRef: RefObject<HTMLElement>
) => {
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
        getInitialData: () => {
          const element = buildNodeFieldElement(fieldIdentifier.nodeId, fieldIdentifier.fieldName, fieldTemplate.type);
          return buildFormElementDndData(element);
        },
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
      })
    );
  }, [dragHandleRef, draggableRef, fieldIdentifier, fieldTemplate.type]);

  return isDragging;
};
