import type { CenterOrEdge } from 'features/nodes/components/sidePanel/builder/center-or-closest-edge';
import type { NodesState } from 'features/nodes/store/types';
import type { StatefulFieldValue } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { BuilderForm, ElementId, FormElement } from 'features/nodes/types/workflow';
import { isContainerElement, isNodeFieldElement } from 'features/nodes/types/workflow';
import { assert } from 'tsafe';
/**
 * Removes an element from the form.
 * The element is removed from its parent. If the element is a container, its children are also recursively removed.
 * The form is mutated in place.
 *
 * @param args.form The form to remove the element from
 * @param args.id The ID of the element to remove
 *
 * @returns True if the element was removed, false otherwise
 */
export const removeElement = (args: { form: BuilderForm; id: string }): boolean => {
  const { id, form } = args;

  const element = form.elements[id];

  // Bail if the element doesn't exist
  if (!element) {
    return false;
  }

  if (form.rootElementId === id || !element.parentId) {
    // Can't remove the root element
    return false;
  }

  const parent = form.elements[element.parentId];
  if (!parent || !isContainerElement(parent)) {
    // This should never happen!
    return false;
  }

  // Recursively remove children if the element is a container
  // TODO(psyche): Handle/guard against circular references?
  if (isContainerElement(element)) {
    for (const childId of [...element.data.children]) {
      removeElement({ form, id: childId });
    }
  }

  delete form.elements[id];
  parent.data.children = parent.data.children.filter((childId) => childId !== id);

  return true;
};

/**
 * Reparents an element in the form.
 * The element is removed from its parent and inserted at the given index in the new parent.
 * The form is mutated in place.
 *
 * @param args.form The form to reparent the element in
 * @param args.id The ID of the element to reparent
 * @param args.newParentId The ID of the new parent element
 * @param args.index The index to insert the element at in the new parent
 *
 * @returns True if the element was reparented, false otherwise
 */
export const reparentElement = (args: {
  form: BuilderForm;
  id: string;
  newParentId: string;
  index: number;
}): boolean => {
  const { form, id, newParentId, index } = args;
  const { elements } = form;

  const element = elements[id];

  // Bail if the element doesn't exist
  if (!element) {
    return false;
  }

  if (form.rootElementId === element.id || !element.parentId) {
    // Can't reparent the root element
    return false;
  }

  if (newParentId === element.parentId) {
    // Nothing to do if the element is already a child of the new parent
    return true;
  }

  const oldParent = elements[element.parentId];
  if (!oldParent || !isContainerElement(oldParent)) {
    // This should never happen
    return false;
  }

  const newParent = elements[newParentId];
  if (!newParent || !isContainerElement(newParent)) {
    return false;
  }

  newParent.data.children.splice(index, 0, id);
  oldParent.data.children = oldParent.data.children.filter((elementId) => elementId !== id);
  element.parentId = newParentId;
  return true;
};

/**
 * Adds an element to the form.
 * The element is added as a child of the given parent at the given index.
 * The form is mutated in place.
 *
 * @param args.form The form to add the element to
 * @param args.element The element to add
 * @param args.index The index to insert the element at in the parent
 *
 * @returns True if the element was added, false otherwise
 */
export const addElement = (args: {
  form: BuilderForm;
  element: FormElement;
  parentId: string;
  index?: number;
}): boolean => {
  const { form, element, parentId, index } = args;
  const { elements } = form;

  if (element.id in elements) {
    // Element already exists
    return false;
  }

  const parent = elements[parentId];
  if (!parent || !isContainerElement(parent)) {
    return false;
  }

  element.parentId = parentId;
  elements[element.id] = element;
  parent.data.children.splice(index ?? parent.data.children.length, 0, element.id);
  return true;
};

/**
 * Checks if an element exists in the form.
 *
 * @param form The form to check
 * @param elementId The id of the element to check
 *
 * @returns True if the element exists, false otherwise
 */
export const elementExists = (form: BuilderForm, elementId: ElementId): boolean => {
  return form.elements[elementId] !== undefined;
};

export type FormElementTypeGuard<T extends FormElement> = (element: FormElement) => element is T;

/**
 * Gets an element from the form.
 *
 * @param form The form to get the element from
 * @param elementId The id of the element to get
 * @param guard An optional guard to ensure the element is of the correct type
 *
 * @returns The element
 *
 * @raises If the element does not exist or the guard is provided and fails
 */
export const getElement = <T extends FormElement>(
  form: BuilderForm,
  elementId: ElementId,
  guard?: FormElementTypeGuard<T>
): T => {
  const el = form.elements[elementId];
  assert(el);
  if (guard) {
    assert(guard(el));
    return el;
  } else {
    return el as T;
  }
};

/**
 * Gets the initial value of a node field element.
 *
 * @param nodes The nodes state to get the initial value from
 * @param element The element to get the initial value for
 *
 * @returns The initial value of the element, if it exists
 */
export const getInitialValue = (nodes: NodesState['nodes'], element: FormElement): StatefulFieldValue => {
  if (!isNodeFieldElement(element)) {
    return undefined;
  }
  const { nodeId, fieldName } = element.data.fieldIdentifier;
  const node = nodes.find((n) => n.id === nodeId);

  // The node or input not existing as we add it to the builder indicates something is wrong, but we don't want to
  // throw an error here as it could break the builder. Can we handle this better?
  if (!node) {
    return undefined;
  }
  if (!isInvocationNode(node)) {
    return undefined;
  }
  const input = node.data.inputs[fieldName];
  if (!input) {
    return undefined;
  }
  return input.value;
};

/**
 * Gets the allowed drop regions for an form element drop target.
 *
 * Containers without children have only one allowed drop region, 'center'. This indicates the element will be added
 * as the first child of the container.
 *
 * In all other cases, elements can be dropped to the 'left' and 'right, or 'top' and 'bottom' edges of the target
 * element. This indicates the element will be inserted in the target's parent container before or after the target.
 *
 * @param form The form
 * @param element The element to get allowed drop regions for
 *
 * @returns The allowed drop regions
 */
export const getAllowedDropRegions = (form: BuilderForm, element: FormElement): CenterOrEdge[] => {
  const dropRegions: CenterOrEdge[] = [];

  if (isContainerElement(element) && element.data.children.length === 0) {
    dropRegions.push('center');
  }

  // Parent is a container
  if (element.parentId !== undefined) {
    const parentContainer = getElement(form, element.parentId, isContainerElement);
    if (parentContainer.data.layout === 'row') {
      dropRegions.push('left', 'right');
    } else {
      // parentContainer.data.layout === 'column'
      dropRegions.push('top', 'bottom');
    }
  }

  return dropRegions;
};

/**
 * Checks if a form is empty.
 * A form is empty if it only contains the root element and the root element has no children.
 * @param form The form to check
 * @returns True if the form is empty, false otherwise
 */
export const getIsFormEmpty = (form: BuilderForm): boolean => {
  const rootElement = form.elements[form.rootElementId];
  if (!rootElement || !isContainerElement(rootElement)) {
    return true;
  }
  return Object.keys(form.elements).length === 1 && rootElement.data.children.length === 0;
};
