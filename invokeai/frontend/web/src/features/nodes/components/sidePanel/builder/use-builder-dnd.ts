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
import { parseify } from 'common/util/serialize';
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
import { useCallback, useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import type { Param0 } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('dnd');

const uniqueRootKey = Symbol('root');
type RootDndData = {
  [uniqueRootKey]: true;
};
const buildRootDndData = (): RootDndData => ({
  [uniqueRootKey]: true,
});
const isRootDndData = (data: Record<string | symbol, unknown>): data is RootDndData => {
  return uniqueRootKey in data;
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

  const dispatchAndFlash = useCallback(
    (action: Param0<typeof dispatch>, elementId: ElementId) => {
      flushSync(() => {
        dispatch(action);
      });
      flashElement(elementId);
    },
    [dispatch]
  );

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

        if (!isFormElementDndData(sourceData)) {
          return;
        }

        const isAddingNewElement = !elementExists(sourceData.element.id);

        //#region Dragging onto the root
        if (isRootDndData(targetData)) {
          if (isAddingNewElement) {
            log.debug('Adding new element to empty root');
            dispatchAndFlash(
              formElementAdded({
                element: sourceData.element,
                containerId: undefined,
                index: undefined,
              }),
              sourceData.element.id
            );
            return;
          }

          if (!isAddingNewElement && sourceData.element.parentId !== undefined) {
            log.debug('Reparenting element from container to empty root');
            dispatchAndFlash(
              formElementReparented({
                id: sourceData.element.id,
                containerId: undefined,
                index: undefined,
              }),
              sourceData.element.id
            );
            return;
          }
        }
        //#endregion

        //#region Form elements
        if (isFormElementDndData(targetData)) {
          const closestEdgeOfTarget = extractClosestCenterOrEdge(targetData);

          const targetElement = getElement(targetData.element.id);
          const sourceElement = getElement(sourceData.element.id);

          if (isAddingNewElement && targetElement.parentId === undefined && closestEdgeOfTarget === 'center') {
            log.debug('Adding new element to empty container');
            assert(isContainerElement(targetElement), 'Expected target to be a container');
            dispatchAndFlash(
              formElementAdded({
                element: sourceElement,
                containerId: targetElement.id,
              }),
              sourceElement.id
            );
            return;
          }

          if (isAddingNewElement && targetElement.parentId === undefined && closestEdgeOfTarget !== 'center') {
            log.debug('Inserting new element into root');
            const layout = getLayout();
            const indexOfTarget = layout.indexOf(targetElement.id);
            const index = getReorderDestinationIndex({
              startIndex: indexOfTarget + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: 'vertical',
            });
            dispatchAndFlash(
              formElementAdded({
                element: sourceElement,
                containerId: undefined,
                index,
              }),
              sourceElement.id
            );
            return;
          }

          if (isAddingNewElement && targetElement.parentId !== undefined && closestEdgeOfTarget === 'center') {
            log.debug('Adding new element to empty container');
            assert(isContainerElement(targetElement), 'Expected target to be a container');
            dispatchAndFlash(
              formElementAdded({
                element: sourceElement,
                containerId: targetElement.id,
                index: undefined,
              }),
              sourceElement.id
            );
            return;
          }

          if (isAddingNewElement && targetElement.parentId !== undefined && closestEdgeOfTarget !== 'center') {
            log.debug('Inserting new element into container');
            const container = getElement(targetElement.parentId, isContainerElement);
            const indexOfTarget = container.data.children.indexOf(targetElement.id);
            const index = getReorderDestinationIndex({
              startIndex: indexOfTarget + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            dispatchAndFlash(
              formElementAdded({
                element: sourceElement,
                containerId: container.id,
                index,
              }),
              sourceElement.id
            );
            return;
          }

          // Moving an existing element within the root
          if (
            !isAddingNewElement &&
            targetElement.parentId === undefined &&
            sourceElement.parentId === undefined &&
            closestEdgeOfTarget === 'center'
          ) {
            log.debug('Reparenting element from root to empty container');
            assert(isContainerElement(targetElement), 'Expected target to be a container');
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: targetElement.id,
                index: undefined,
              }),
              sourceElement.id
            );
            return;
          }
          if (
            !isAddingNewElement &&
            targetElement.parentId === undefined &&
            sourceElement.parentId === undefined &&
            closestEdgeOfTarget !== 'center'
          ) {
            log.debug('Moving element within root');
            const layout = getLayout();
            const startIndex = layout.indexOf(sourceElement.id);
            const indexOfTarget = layout.indexOf(targetElement.id);
            const reorderedLayout = reorderWithEdge({
              list: layout,
              startIndex,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: 'vertical',
            });
            dispatchAndFlash(
              formRootReordered({
                layout: reorderedLayout,
              }),
              sourceElement.id
            );
            return;
          }

          // Moving an existing element within a container
          if (
            !isAddingNewElement &&
            targetElement.parentId !== undefined &&
            sourceElement.parentId !== undefined &&
            targetElement.parentId === sourceElement.parentId &&
            closestEdgeOfTarget === 'center'
          ) {
            log.debug('Reparenting element from a container to an empty container with same parent');
            assert(isContainerElement(targetElement), 'Expected target to be a container');
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: targetElement.id,
                index: undefined,
              }),
              sourceElement.id
            );
            return;
          }

          if (
            !isAddingNewElement &&
            targetElement.parentId !== undefined &&
            sourceElement.parentId !== undefined &&
            targetElement.parentId === sourceElement.parentId &&
            closestEdgeOfTarget !== 'center'
          ) {
            log.debug('Moving element within container');
            const container = getElement(targetElement.parentId, isContainerElement);
            const startIndex = container.data.children.indexOf(sourceElement.id);
            const indexOfTarget = container.data.children.indexOf(targetElement.id);
            const reorderedLayout = reorderWithEdge({
              list: container.data.children,
              startIndex,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            dispatchAndFlash(
              formContainerChildrenReordered({
                containerId: container.id,
                children: reorderedLayout,
              }),
              sourceElement.id
            );
            return;
          }

          // Moving an existing element within a container
          if (
            !isAddingNewElement &&
            targetElement.parentId !== undefined &&
            sourceElement.parentId !== undefined &&
            targetElement.parentId !== sourceElement.parentId &&
            closestEdgeOfTarget === 'center'
          ) {
            log.debug('Reparenting element from one container to an empty container with different parent');
            assert(isContainerElement(targetElement), 'Expected target to be a container');
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: targetElement.id,
                index: undefined,
              }),
              sourceElement.id
            );
            return;
          }

          if (
            !isAddingNewElement &&
            targetElement.parentId !== undefined &&
            sourceElement.parentId !== undefined &&
            targetElement.parentId !== sourceElement.parentId &&
            closestEdgeOfTarget !== 'center'
          ) {
            log.debug('Moving element from one container to another');
            const container = getElement(targetElement.parentId, isContainerElement);
            const indexOfTarget = container.data.children.indexOf(targetElement.id);
            const index = getReorderDestinationIndex({
              startIndex: container.data.children.length + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: container.id,
                index,
              }),
              sourceElement.id
            );
            return;
          }

          // Moving an existing element from a container to root
          if (
            !isAddingNewElement &&
            targetElement.parentId === undefined &&
            sourceElement.parentId !== undefined &&
            closestEdgeOfTarget === 'center'
          ) {
            log.debug('Reparenting element from container to empty container');
            assert(isContainerElement(targetElement), 'Expected target to be a container');
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: targetElement.id,
                index: undefined,
              }),
              sourceElement.id
            );
            return;
          }
          if (
            !isAddingNewElement &&
            targetElement.parentId === undefined &&
            sourceElement.parentId !== undefined &&
            closestEdgeOfTarget !== 'center'
          ) {
            log.debug('Reparenting element from container to root');
            const layout = getLayout();
            const indexOfTarget = layout.indexOf(targetElement.id);
            const index = getReorderDestinationIndex({
              startIndex: layout.length + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: 'vertical',
            });
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: undefined,
                index,
              }),
              sourceElement.id
            );
            return;
          }

          // Moving an existing element from root to a container
          if (
            !isAddingNewElement &&
            targetElement.parentId !== undefined &&
            sourceElement.parentId === undefined &&
            closestEdgeOfTarget === 'center'
          ) {
            log.debug('Reparenting element from root to empty container');
            assert(isContainerElement(targetElement), 'Expected target to be a container');
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: targetElement.id,
                index: undefined,
              }),
              sourceElement.id
            );
            return;
          }

          if (
            !isAddingNewElement &&
            targetElement.parentId !== undefined &&
            sourceElement.parentId === undefined &&
            closestEdgeOfTarget !== 'center'
          ) {
            log.debug('Reparenting element from root to container');
            const container = getElement(targetElement.parentId, isContainerElement);
            const indexOfTarget = container.data.children.indexOf(targetElement.id);
            const index = getReorderDestinationIndex({
              startIndex: indexOfTarget + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                containerId: targetElement.parentId,
                index,
              }),
              sourceElement.id
            );
            return;
          }
        }
        //#endregion

        log.warn(parseify({ targetData, sourceData }), 'Unhandled drop event!');
      },
    });
  }, [dispatchAndFlash]);
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
      getData: () => buildRootDndData(),
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
        getIsSticky: () => getElement(elementId).parentId === undefined,
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
