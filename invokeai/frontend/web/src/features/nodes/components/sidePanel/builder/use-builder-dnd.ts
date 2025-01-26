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
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import { formElementAdded, formElementMoved } from 'features/nodes/store/workflowSlice';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { ElementId, FormElement } from 'features/nodes/types/workflow';
import { buildNodeField, isContainerElement } from 'features/nodes/types/workflow';
import type { RefObject } from 'react';
import { useCallback, useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { assert } from 'tsafe';

const uniqueMoveFormElementKey = Symbol('move-form-element');
type MoveFormElementDndData = {
  [uniqueMoveFormElementKey]: true;
  element: FormElement;
};
const buildMoveFormElementDndData = (element: FormElement): MoveFormElementDndData => ({
  [uniqueMoveFormElementKey]: true,
  element,
});
const isMoveFormElementDndData = (data: Record<string | symbol, unknown>): data is MoveFormElementDndData => {
  return uniqueMoveFormElementKey in data;
};

const uniqueAddFormElementKey = Symbol('add-form-element');
type AddFormElementDndData = {
  [uniqueAddFormElementKey]: true;
  element: FormElement;
};
export const buildAddFormElementDndData = (element: FormElement): AddFormElementDndData => ({
  [uniqueAddFormElementKey]: true,
  element,
});
const isAddFormElementDndData = (data: Record<string | symbol, unknown>): data is AddFormElementDndData => {
  return uniqueAddFormElementKey in data;
};

const uniqueNodeFieldKey = Symbol('node-field');
type NodeFieldDndData = {
  [uniqueNodeFieldKey]: true;
  fieldIdentifier: FieldIdentifier;
};
export const buildNodeFieldDndData = (fieldIdentifier: FieldIdentifier): NodeFieldDndData => ({
  [uniqueNodeFieldKey]: true,
  fieldIdentifier,
});

const isNodeFieldDndData = (data: Record<string | symbol, unknown>): data is NodeFieldDndData => {
  return uniqueNodeFieldKey in data;
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

const adjustIndexForFormElementMoveDrop = (index: number, edge: Exclude<CenterOrEdge, 'center'>) => {
  if (edge === 'left' || edge === 'top') {
    return index - 1;
  }
  return index + 1;
};

const adjustIndexForNodeFieldDrop = (index: number, edge: Exclude<CenterOrEdge, 'center'>) => {
  if (edge === 'left' || edge === 'top') {
    return index;
  }
  return index + 1;
};

const flashElement = (elementId: ElementId) => {
  const element = document.querySelector(`#${getEditModeWrapperId(elementId)}`);
  if (element instanceof HTMLElement) {
    triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
  }
};

export const useMonitorForFormElementDnd = () => {
  const dispatch = useAppDispatch();

  const handleMoveFormElementDrop = useCallback(
    (sourceData: MoveFormElementDndData, targetData: MoveFormElementDndData) => {
      if (sourceData.element.id === targetData.element.id) {
        return;
      }

      const closestCenterOrEdge = extractClosestCenterOrEdge(targetData);

      if (closestCenterOrEdge === 'center') {
        // Move the element to the target container - should we double-check that the target is a container?
        flushSync(() => {
          dispatch(formElementMoved({ id: sourceData.element.id, containerId: targetData.element.id }));
        });
        // Flash the element that was moved
        flashElement(sourceData.element.id);
      } else if (closestCenterOrEdge) {
        // Move the element to the target's parent container at the correct index
        const { parentId } = targetData.element;
        assert(parentId !== undefined, 'Target element should have a parent');

        const isReparenting = parentId !== sourceData.element.parentId;

        const parentContainer = getElement(parentId, isContainerElement);
        const targetIndex = parentContainer.data.children.findIndex((elementId) => elementId === targetData.element.id);

        let index: number | undefined = undefined;

        if (!isReparenting) {
          const sourceIndex = parentContainer.data.children.findIndex(
            (elementId) => elementId === sourceData.element.id
          );
          if (
            sourceIndex === targetIndex ||
            sourceIndex === adjustIndexForFormElementMoveDrop(targetIndex, closestCenterOrEdge)
          ) {
            return;
          }
          index = targetIndex;
        } else {
          index = adjustIndexForFormElementMoveDrop(targetIndex, closestCenterOrEdge);
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
        // Flash the element that was moved
        flashElement(sourceData.element.id);
      } else {
        // No container, cannot do anything
        return;
      }
    },
    [dispatch]
  );

  const handleAddFormElementDrop = useCallback(
    (sourceData: AddFormElementDndData, targetData: MoveFormElementDndData) => {
      const closestCenterOrEdge = extractClosestCenterOrEdge(targetData);

      if (closestCenterOrEdge === 'center') {
        // Move the element to the target container - should we double-check that the target is a container?
        const { element } = sourceData;
        flushSync(() => {
          dispatch(formElementAdded({ element, containerId: targetData.element.id }));
        });
        flashElement(element.id);
      } else if (closestCenterOrEdge) {
        // Move the element to the target's parent container at the correct index
        const { parentId } = targetData.element;
        assert(parentId !== undefined, 'Target element should have a parent');
        const { element } = sourceData;

        const parentContainer = getElement(parentId, isContainerElement);
        const targetIndex = parentContainer.data.children.findIndex((elementId) => elementId === targetData.element.id);

        const index = adjustIndexForNodeFieldDrop(targetIndex, closestCenterOrEdge);

        flushSync(() => {
          dispatch(
            formElementAdded({
              element,
              containerId: parentId,
              index,
            })
          );
        });
        flashElement(element.id);
      } else {
        // No container, cannot do anything
        return;
      }
    },
    [dispatch]
  );

  const handleNodeFieldDrop = useCallback(
    (sourceData: NodeFieldDndData, targetData: MoveFormElementDndData) => {
      const closestCenterOrEdge = extractClosestCenterOrEdge(targetData);
      const { nodeId, fieldName } = sourceData.fieldIdentifier;

      if (closestCenterOrEdge === 'center') {
        // Move the element to the target container - should we double-check that the target is a container?
        const element = buildNodeField(nodeId, fieldName, targetData.element.id);
        flushSync(() => {
          dispatch(formElementAdded({ element, containerId: targetData.element.id }));
        });
        flashElement(element.id);
      } else if (closestCenterOrEdge) {
        // Move the element to the target's parent container at the correct index
        const { parentId } = targetData.element;
        assert(parentId !== undefined, 'Target element should have a parent');
        const element = buildNodeField(nodeId, fieldName, parentId);

        const parentContainer = getElement(parentId, isContainerElement);
        const targetIndex = parentContainer.data.children.findIndex((elementId) => elementId === targetData.element.id);

        const index = adjustIndexForNodeFieldDrop(targetIndex, closestCenterOrEdge);

        flushSync(() => {
          dispatch(
            formElementAdded({
              element,
              containerId: parentId,
              index,
            })
          );
        });
        flashElement(element.id);
      } else {
        // No container, cannot do anything
        return;
      }
    },
    [dispatch]
  );

  useEffect(() => {
    return monitorForElements({
      canMonitor: ({ source }) =>
        isMoveFormElementDndData(source.data) ||
        isNodeFieldDndData(source.data) ||
        isAddFormElementDndData(source.data),
      onDrop: ({ location, source }) => {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceData = source.data;
        const targetData = target.data;

        if (isMoveFormElementDndData(targetData) && isMoveFormElementDndData(sourceData)) {
          handleMoveFormElementDrop(sourceData, targetData);
          return;
        }

        if (isMoveFormElementDndData(targetData) && isAddFormElementDndData(sourceData)) {
          handleAddFormElementDrop(sourceData, targetData);
          return;
        }

        if (isMoveFormElementDndData(targetData) && isNodeFieldDndData(sourceData)) {
          handleNodeFieldDrop(sourceData, targetData);
          return;
        }
      },
    });
  }, [handleAddFormElementDrop, handleMoveFormElementDrop, handleNodeFieldDrop]);
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
        // The root element is not draggable
        canDrag: () => Boolean(_element.parentId),
        element: draggableElement,
        dragHandle: dragHandleElement,
        getInitialData: () => buildMoveFormElementDndData(getElement(elementId)),
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
          isMoveFormElementDndData(source.data) ||
          isNodeFieldDndData(source.data) ||
          isAddFormElementDndData(source.data),
        getData: ({ input }) => {
          const element = getElement(elementId);
          const container = element.parentId ? getElement(element.parentId, isContainerElement) : null;

          const data = buildMoveFormElementDndData(element);

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
