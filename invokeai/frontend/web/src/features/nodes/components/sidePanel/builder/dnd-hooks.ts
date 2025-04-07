import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import type { DropTargetRecord } from '@atlaskit/pragmatic-drag-and-drop/dist/types/internal-types';
import type { ElementDragPayload } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import {
  draggable,
  dropTargetForElements,
  monitorForElements,
} from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { getReorderDestinationIndex } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/get-reorder-destination-index';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { parseify } from 'common/util/serialize';
import { firefoxDndFix, triggerPostMoveFlash } from 'features/dnd/util';
import type { CenterOrEdge } from 'features/nodes/components/sidePanel/builder/center-or-closest-edge';
import {
  attachClosestCenterOrEdge,
  extractClosestCenterOrEdge,
} from 'features/nodes/components/sidePanel/builder/center-or-closest-edge';
import type { FormElementTypeGuard } from 'features/nodes/components/sidePanel/builder/form-manipulation';
import {
  elementExists,
  getAllowedDropRegions,
  getElement,
  getInitialValue,
} from 'features/nodes/components/sidePanel/builder/form-manipulation';
import {
  formElementAdded,
  formElementContainerDataChanged,
  formElementReparented,
} from 'features/nodes/store/nodesSlice';
import { selectFormRootElementId, selectNodesSlice, selectWorkflowForm } from 'features/nodes/store/selectors';
import type { FieldInputTemplate, StatefulFieldValue } from 'features/nodes/types/field';
import type { ElementId, FormElement } from 'features/nodes/types/workflow';
import { buildNodeFieldElement, isContainerElement } from 'features/nodes/types/workflow';
import type { RefObject } from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { flushSync } from 'react-dom';
import type { Param0 } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('dnd');

// Dnd payloads are arbitrarily shaped. We can use a unique symbol as a sentinel value for our strongly-typed dnd data.
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

const uniqueNodeFieldDndKey = Symbol('node-field');
type NodeFieldDndData = {
  [uniqueNodeFieldDndKey]: true;
  nodeId: string;
  fieldName: string;
  fieldTemplate: FieldInputTemplate;
};
const buildNodeFieldDndData = (
  nodeId: string,
  fieldName: string,
  fieldTemplate: FieldInputTemplate
): NodeFieldDndData => ({
  [uniqueNodeFieldDndKey]: true,
  nodeId,
  fieldName,
  fieldTemplate,
});
const isNodeFieldDndData = (data: Record<string | symbol, unknown>): data is NodeFieldDndData => {
  return uniqueNodeFieldDndKey in data;
};

/**
 * Flashes an element by changing its background color. Used to indicate that an element has been moved.
 * @param elementId The id of the element to flash
 */
const flashElement = (elementId: ElementId) => {
  const element = document.querySelector(`#${elementId}`);
  if (element instanceof HTMLElement) {
    triggerPostMoveFlash(element, colorTokenToCssVar('base.800'));
  }
};

/**
 * Wrapper around `getElement` that provides the form state from the store.
 * @see {@link getElement}
 */
const useGetElement = () => {
  const store = useAppStore();
  const _getElement = useCallback(
    <T extends FormElement>(elementId: ElementId, guard?: FormElementTypeGuard<T>): T => {
      const form = selectWorkflowForm(store.getState());
      return getElement(form, elementId, guard);
    },
    [store]
  );
  return _getElement;
};

/**
 * Wrapper around `elementExists` that provides the form state from the store.
 * @see {@link elementExists}
 */
const useElementExists = () => {
  const store = useAppStore();
  const _elementExists = useCallback(
    (id: ElementId): boolean => {
      const form = selectWorkflowForm(store.getState());
      return elementExists(form, id);
    },
    [store]
  );
  return _elementExists;
};

/**
 * Wrapper around `getAllowedDropRegions` that provides the form state from the store.
 * @see {@link getAllowedDropRegions}
 */
const useGetAllowedDropRegions = () => {
  const store = useAppStore();
  const _getAllowedDropRegions = useCallback(
    (element: FormElement): CenterOrEdge[] => {
      const form = selectWorkflowForm(store.getState());
      return getAllowedDropRegions(form, element);
    },
    [store]
  );
  return _getAllowedDropRegions;
};

/**
 * Wrapper around `getInitialValue` that provides the nodes state from the store.
 * @see {@link getInitialValue}
 */
const useGetInitialValue = () => {
  const store = useAppStore();
  const _getInitialValue = useCallback(
    (element: FormElement): StatefulFieldValue => {
      const { nodes } = selectNodesSlice(store.getState());
      return getInitialValue(nodes, element);
    },
    [store]
  );
  return _getInitialValue;
};

const getSourceElement = (source: ElementDragPayload) => {
  if (isNodeFieldDndData(source.data)) {
    const { nodeId, fieldName, fieldTemplate } = source.data;
    return buildNodeFieldElement(nodeId, fieldName, fieldTemplate.type);
  }

  if (isFormElementDndData(source.data)) {
    return source.data.element;
  }

  return null;
};

const getTargetElement = (target: DropTargetRecord) => {
  if (isFormElementDndData(target.data)) {
    return target.data.element;
  }

  return null;
};

/**
 * Singleton hook that monitors for builder dnd events and dispatches actions accordingly.
 */
export const useBuilderDndMonitor = () => {
  useAssertSingleton('useBuilderDndMonitor');
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

  const getElement = useGetElement();
  const getInitialValue = useGetInitialValue();
  const elementExists = useElementExists();

  useEffect(() => {
    return monitorForElements({
      canMonitor: ({ source }) => isFormElementDndData(source.data) || isNodeFieldDndData(source.data),
      onDrop: ({ location, source }) => {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceElement = getSourceElement(source);
        const targetElement = getTargetElement(target);

        if (!sourceElement || !targetElement) {
          return;
        }

        if (sourceElement.id === targetElement.id) {
          // Dropping on self is a no-op
          return;
        }

        const isAddingNewElement = !elementExists(sourceElement.id);
        const closestEdgeOfTarget = extractClosestCenterOrEdge(target.data);

        /**
         * There are 5 cases to handle:
         * 1. Adding a new element to an empty container
         * 2. Adding a new element to a container with children (dropped on edge of container child)
         * 3. Reparenting an element to an empty container
         * 4. Reparenting an element to a container with children (dropped on edge of container child)
         * 5. Moving an element within a container
         *
         * We can determine which case we're in by checking the following:
         * - Check if the element already exists in the form. If it doesn't, we're adding a new element.
         * - Check if the closest edge of the target is 'center'. If it is, we're either adding a new element or reparenting to an empty container.
         * - If the closest edge of the target is not 'center', we're either reparenting to a container with children or moving an element within a container, in which case we compare the parent of the source and target elements.
         *
         */

        if (isAddingNewElement) {
          if (closestEdgeOfTarget === 'center') {
            log.trace('Adding new element to empty container');
            if (!isContainerElement(targetElement)) {
              log.error(parseify({ target, source }), 'Expected target to be a container');
              return;
            }
            dispatchAndFlash(
              formElementAdded({
                element: sourceElement,
                parentId: targetElement.id,
                index: 0,
                initialValue: getInitialValue(sourceElement),
              }),
              sourceElement.id
            );
            return;
          } else {
            log.trace('Adding new element to container with children (dropped on edge of container child)');
            if (targetElement.parentId === undefined) {
              log.error(parseify({ target, source }), 'Expected target to have a parent');
              return;
            }
            const parent = getElement(targetElement.parentId, isContainerElement);
            const indexOfTarget = parent.data.children.indexOf(targetElement.id);
            const index = getReorderDestinationIndex({
              startIndex: indexOfTarget + 1,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: parent.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            dispatchAndFlash(
              formElementAdded({
                element: sourceElement,
                parentId: parent.id,
                index,
                initialValue: getInitialValue(sourceElement),
              }),
              sourceElement.id
            );
            return;
          }
        } else {
          if (closestEdgeOfTarget === 'center') {
            log.trace('Reparenting element to an empty container');
            if (!isContainerElement(targetElement)) {
              log.error(parseify({ target, source }), 'Expected target to be a container');
              return;
            }
            dispatchAndFlash(
              formElementReparented({
                id: sourceElement.id,
                newParentId: targetElement.id,
                index: 0,
              }),
              sourceElement.id
            );
            return;
          } else if (targetElement.parentId === sourceElement.parentId) {
            log.trace('Moving element within container');
            if (targetElement.parentId === undefined) {
              log.error(parseify({ target, source }), 'Expected target to have a parent');
              return;
            }
            const container = getElement(targetElement.parentId, isContainerElement);
            const startIndex = container.data.children.indexOf(sourceElement.id);
            const indexOfTarget = container.data.children.indexOf(targetElement.id);
            const reorderedChildren = reorderWithEdge({
              list: container.data.children,
              startIndex,
              indexOfTarget,
              closestEdgeOfTarget,
              axis: container.data.layout === 'row' ? 'horizontal' : 'vertical',
            });
            dispatchAndFlash(
              formElementContainerDataChanged({
                id: container.id,
                changes: { children: reorderedChildren },
              }),
              sourceElement.id
            );
            return;
          } else if (targetElement.parentId !== sourceElement.parentId) {
            log.trace('Reparenting element to container with children (dropped on edge of container child)');
            if (targetElement.parentId === undefined) {
              log.error(parseify({ target, source }), 'Expected target to have a parent');
              return;
            }
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
                newParentId: container.id,
                index,
              }),
              sourceElement.id
            );
            return;
          } else {
            // This should never happen
            log.error(parseify({ target, source }), 'Unhandled drop event!');
          }
        }
      },
    });
  }, [dispatchAndFlash, elementExists, getElement, getInitialValue]);
};

/**
 * Hook that provides dnd functionality for builder form elements.
 *
 * @param elementId The id of the form element
 * @param draggableRef The ref of the draggable HTML element
 * @param dragHandleRef The ref of the drag handle HTML element
 *
 * @returns A tuple containing the active drop region and whether the element is currently being dragged
 */
export const useFormElementDnd = (
  elementId: ElementId,
  draggableRef: RefObject<HTMLElement>,
  dragHandleRef: RefObject<HTMLElement>
) => {
  const isRootElement = useIsRootElement(elementId);
  const [isDragging, setIsDragging] = useState(false);
  const [activeDropRegion, setActiveDropRegion] = useState<CenterOrEdge | null>(null);
  const getElement = useGetElement();
  const getAllowedDropRegions = useGetAllowedDropRegions();

  useEffect(() => {
    if (isRootElement) {
      assert(false, 'Root element should not be draggable');
    }
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
          const element = getElement(elementId);
          return buildFormElementDndData(element);
        },
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
      }),
      dropTargetForElements({
        element: draggableElement,
        // TODO(psyche): This causes a kinda jittery behaviour - need a better heuristic to determine stickiness
        getIsSticky: () => false,
        canDrop: ({ source }) => {
          if (isNodeFieldDndData(source.data)) {
            return true;
          }
          if (isFormElementDndData(source.data)) {
            return source.data.element.id !== getElement(elementId).parentId;
          }
          return false;
        },
        getData: ({ input }) => {
          const element = getElement(elementId);

          const targetData = buildFormElementDndData(element);

          const allowedCenterOrEdge = getAllowedDropRegions(element);

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

          // Don't allow reparenting to the same container
          if (closestCenterOrEdge === 'center' && source.element === draggableElement) {
            setActiveDropRegion(null);
            return;
          }

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
  }, [dragHandleRef, draggableRef, elementId, getAllowedDropRegions, getElement, isRootElement]);

  return [activeDropRegion, isDragging] as const;
};

export const useRootElementDropTarget = (droppableRef: RefObject<HTMLDivElement>) => {
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const getElement = useGetElement();
  const getAllowedDropRegions = useGetAllowedDropRegions();
  const rootElementId = useAppSelector(selectFormRootElementId);

  useEffect(() => {
    const droppableElement = droppableRef.current;

    if (!droppableElement) {
      return;
    }

    return combine(
      dropTargetForElements({
        element: droppableElement,
        getIsSticky: () => false,
        canDrop: ({ source }) => {
          const rootElement = getElement(rootElementId, isContainerElement);
          if (rootElement.data.children.length !== 0) {
            return false;
          }
          if (isNodeFieldDndData(source.data) || isFormElementDndData(source.data)) {
            return true;
          }
          return false;
        },
        getData: ({ input }) => {
          const element = getElement(rootElementId, isContainerElement);

          const targetData = buildFormElementDndData(element);

          return attachClosestCenterOrEdge(targetData, {
            element: droppableElement,
            input,
            allowedCenterOrEdge: ['center'],
          });
        },
        onDrag: () => {
          setIsDraggingOver(true);
        },
        onDragLeave: () => {
          setIsDraggingOver(false);
        },
        onDrop: () => {
          setIsDraggingOver(false);
        },
      })
    );
  }, [droppableRef, getAllowedDropRegions, getElement, rootElementId]);

  return isDraggingOver;
};

/**
 * Hook that provides dnd functionality for node fields.
 *
 * @param nodeId: The id of the node
 * @param fieldName: The name of the field
 * @param fieldTemplate The template of the node field, required to build the form element
 * @param draggableRef The ref of the draggable HTML element
 * @param dragHandleRef The ref of the drag handle HTML element
 *
 * @returns Whether the node field is currently being dragged
 */
export const useNodeFieldDnd = (
  nodeId: string,
  fieldName: string,
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
        getInitialData: () => buildNodeFieldDndData(nodeId, fieldName, fieldTemplate),
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
      })
    );
  }, [dragHandleRef, draggableRef, fieldName, fieldTemplate, nodeId]);

  return isDragging;
};

/**
 * Hook that returns whether an element is the root element.
 * @param elementId The id of the element
 * @returns Whether the element is the root element
 */
const useIsRootElement = (elementId: string) => {
  const rootElementId = useAppSelector(selectFormRootElementId);
  const isRootElement = useMemo(() => rootElementId === elementId, [rootElementId, elementId]);
  return isRootElement;
};
