import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
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
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import {
  formElementAdded,
  formElementContainerDataChanged,
  formElementReparented,
  selectFormRootElementId,
  selectWorkflowSlice,
} from 'features/nodes/store/workflowSlice';
import type { FieldIdentifier, FieldInputTemplate, StatefulFieldValue } from 'features/nodes/types/field';
import type { ElementId, FormElement } from 'features/nodes/types/workflow';
import { buildNodeFieldElement, isContainerElement } from 'features/nodes/types/workflow';
import type { RefObject } from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { flushSync } from 'react-dom';
import type { Param0 } from 'tsafe';

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

/**
 * Flashes an element by changing its background color. Used to indicate that an element has been moved.
 * @param elementId The id of the element to flash
 */
const flashElement = (elementId: ElementId) => {
  const element = document.querySelector(`#${getEditModeWrapperId(elementId)}`);
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
      const { form } = selectWorkflowSlice(store.getState());
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
      const { form } = selectWorkflowSlice(store.getState());
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
      const { form } = selectWorkflowSlice(store.getState());
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
      canMonitor: ({ source }) => isFormElementDndData(source.data),
      onDrop: ({ location, source }) => {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        if (!isFormElementDndData(source.data) || !isFormElementDndData(target.data)) {
          return;
        }

        const sourceElement = source.data.element;
        const targetElement = target.data.element;

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
    const draggableElement = draggableRef.current;
    const dragHandleElement = dragHandleRef.current;

    if (!draggableElement || !dragHandleElement) {
      return;
    }

    return combine(
      firefoxDndFix(draggableElement),
      draggable({
        // Don't allow dragging the root element
        canDrag: () => !isRootElement,
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
        getIsSticky: () => !isRootElement,
        canDrop: ({ source }) =>
          isFormElementDndData(source.data) && source.data.element.id !== getElement(elementId).parentId,
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

/**
 * Hook that provides dnd functionality for node fields.
 *
 * @param fieldIdentifier The identifier of the node field
 * @param fieldTemplate The template of the node field, required to build the form element
 * @param draggableRef The ref of the draggable HTML element
 * @param dragHandleRef The ref of the drag handle HTML element
 *
 * @returns Whether the node field is currently being dragged
 */
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
          const { nodeId, fieldName } = fieldIdentifier;
          const { type } = fieldTemplate;
          const element = buildNodeFieldElement(nodeId, fieldName, type);
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
  }, [dragHandleRef, draggableRef, fieldIdentifier, fieldTemplate]);

  return isDragging;
};

/**
 * Hook that returns whether an element is the root element.
 * @param elementId The id of the element
 * @returns Whether the element is the root element
 */
export const useIsRootElement = (elementId: string) => {
  const rootElementId = useAppSelector(selectFormRootElementId);
  const isRootElement = useMemo(() => rootElementId === elementId, [rootElementId, elementId]);
  return isRootElement;
};
