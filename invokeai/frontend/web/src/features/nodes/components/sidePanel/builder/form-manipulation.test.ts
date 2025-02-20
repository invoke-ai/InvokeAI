import { deepClone } from 'common/util/deepClone';
import {
  addElement,
  elementExists,
  getAllowedDropRegions,
  getElement,
  getIsFormEmpty,
  removeElement,
  reparentElement,
} from 'features/nodes/components/sidePanel/builder/form-manipulation';
import type { BuilderForm, ContainerElement } from 'features/nodes/types/workflow';
import {
  buildContainer,
  buildText,
  getDefaultForm,
  isContainerElement,
  isNodeFieldElement,
  validateFormStructure,
} from 'features/nodes/types/workflow';
import type { Equals } from 'tsafe';
import { assert, AssertionError } from 'tsafe';
import { describe, expect, it } from 'vitest';

describe('workflow builder form manipulation', () => {
  describe('elementExists', () => {
    const form = getDefaultForm();

    it('should return true if the element exists', () => {
      expect(elementExists(form, form.rootElementId)).toBe(true);
    });

    it('should return false if the element does not exist', () => {
      expect(elementExists(form, 'non-existent-id')).toBe(false);
    });
  });

  describe('getElement', () => {
    const form = getDefaultForm();

    it('should return the element with the given ID', () => {
      const element = getElement(form, form.rootElementId);
      expect(element).toBe(form.elements[form.rootElementId]);
    });

    it('should return raise if the element does not exist', () => {
      expect(() => getElement(form, 'non-existent-id')).toThrow(AssertionError);
    });

    it('should raise if a type guard is provided and the element does not match', () => {
      expect(() => getElement(form, form.rootElementId, isNodeFieldElement)).toThrow(AssertionError);
    });

    it('should narrow the type of the element if a type guard is provided and the element matches', () => {
      const element = getElement(form, form.rootElementId, isContainerElement);
      assert<Equals<typeof element, ContainerElement>>();
    });
  });

  describe('addElement', () => {
    it('should add the element to the form', () => {
      const form = getDefaultForm();

      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });

      expect(getElement(form, el.id)).toBe(el);
    });

    it('should add the element to the specified parent', () => {
      const form = getDefaultForm();

      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });

      expect(getElement(form, el.id).parentId).toBe(form.rootElementId);
    });

    it('should add the element at the given index', () => {
      const form = getDefaultForm();

      const el1 = buildText('foo');
      addElement({
        form,
        element: el1,
        parentId: form.rootElementId,
        index: 0,
      });

      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el1.id);

      const el2 = buildText('foo');
      addElement({
        form,
        element: el2,
        parentId: form.rootElementId,
        index: 0,
      });

      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el2.id);
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[1]).toBe(el1.id);
    });

    it('should add the element to the end if the index is out of bounds', () => {
      const form = getDefaultForm();
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 100,
      });
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el.id);
    });

    it('should add the element to the end if the index is undefined', () => {
      const form = getDefaultForm();
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
      });
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el.id);
      const el2 = buildText('foo');
      addElement({
        form,
        element: el2,
        parentId: form.rootElementId,
      });
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el.id);
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[1]).toBe(el2.id);
    });

    it('should add the element counting from the end if the index is negative', () => {
      const form = getDefaultForm();
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: -1,
      });
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el.id);
      const el2 = buildText('foo');
      addElement({
        form,
        element: el2,
        parentId: form.rootElementId,
        index: -1,
      });
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el2.id);
      expect(getElement(form, form.rootElementId, isContainerElement).data.children[1]).toBe(el.id);
    });

    it('should return true if the element was added', () => {
      const form = getDefaultForm();

      const el = buildText('foo');

      expect(
        addElement({
          form,
          element: el,
          parentId: form.rootElementId,
          index: 0,
        })
      ).toBe(true);
    });

    it('should noop and return false if the element already exists', () => {
      const form = getDefaultForm();

      const el = buildText('foo');

      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });

      const prevForm = deepClone(form);

      expect(
        addElement({
          form,
          element: el,
          parentId: form.rootElementId,
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it('should noop and return false if the parent does not exist or is not a container', () => {
      const form = getDefaultForm();

      const el = buildText('foo');

      const prevForm = deepClone(form);

      expect(
        addElement({
          form,
          element: el,
          parentId: 'non-existent-id',
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });
  });

  describe('removeElement', () => {
    it('should remove the element from the form', () => {
      const form = getDefaultForm();

      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });

      expect(elementExists(form, el.id)).toBe(true);

      expect(removeElement({ form, id: el.id })).toBe(true);
      expect(elementExists(form, el.id)).toBe(false);
    });

    it('should remove the element from its parent if it has a parent', () => {
      const form = getDefaultForm();

      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });

      expect(getElement(form, form.rootElementId, isContainerElement).data.children[0]).toBe(el.id);

      removeElement({ form, id: el.id });

      expect(getElement(form, form.rootElementId, isContainerElement).data.children).toEqual([]);
    });

    it('should recursively remove the element and its children', () => {
      const form = getDefaultForm();

      const el1 = buildContainer('row', []);
      addElement({
        form,
        element: el1,
        parentId: form.rootElementId,
        index: 0,
      });

      const el2 = buildContainer('row', []);
      addElement({
        form,
        element: el2,
        parentId: el1.id,
        index: 0,
      });

      const el3 = buildText('foo');
      addElement({
        form,
        element: el3,
        parentId: el2.id,
        index: 0,
      });

      expect(elementExists(form, el1.id)).toBe(true);
      expect(elementExists(form, el2.id)).toBe(true);
      expect(elementExists(form, el3.id)).toBe(true);

      removeElement({ form, id: el1.id });

      expect(elementExists(form, el1.id)).toBe(false);
      expect(elementExists(form, el2.id)).toBe(false);
      expect(elementExists(form, el3.id)).toBe(false);
    });

    it('should return true if the element was removed', () => {
      const form = getDefaultForm();

      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });

      expect(removeElement({ form, id: el.id })).toBe(true);
    });

    it('should noop and return false if the element does not exist', () => {
      const form = getDefaultForm();

      const prevForm = deepClone(form);

      expect(removeElement({ form, id: 'non-existent-id' })).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it('should noop and return false if the element is the root element', () => {
      const form = getDefaultForm();

      const prevForm = deepClone(form);

      expect(removeElement({ form, id: form.rootElementId })).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it("should noop and return false if the element's parent does not exist", () => {
      const form = getDefaultForm();

      const el = buildContainer('row', []);
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });

      el.parentId = 'non-existent-id';

      const prevForm = deepClone(form);

      expect(removeElement({ form, id: el.id })).toBe(false);

      expect(form).toEqual(prevForm);
    });
  });

  describe('getAllowedDropRegions', () => {
    it("should return only ['center'] if the target is the root container element and is empty", () => {
      const form = getDefaultForm();
      const rootElement = getElement(form, form.rootElementId);
      expect(getAllowedDropRegions(form, rootElement)).toEqual(['center']);
    });

    it("should return ['center', 'left' and 'right'] if the target is an empty non-root container and its parent has a row layout", () => {
      const form = getDefaultForm();
      const container = buildContainer('row', []);
      addElement({
        form,
        element: container,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildContainer('row', []);
      addElement({
        form,
        element: el,
        parentId: container.id,
        index: 0,
      });
      expect(getAllowedDropRegions(form, el)).toEqual(['center', 'left', 'right']);
    });

    it("should return ['center', 'top' and 'bottom'] if the target is an empty non-root container and its parent has a column layout", () => {
      const form = getDefaultForm();
      const container = buildContainer('column', []);
      addElement({
        form,
        element: container,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildContainer('row', []);
      addElement({
        form,
        element: el,
        parentId: container.id,
        index: 0,
      });
      expect(getAllowedDropRegions(form, el)).toEqual(['center', 'top', 'bottom']);
    });

    it('should return [] when the target is a container with no parent (i.e. it is the root)', () => {
      const form = getDefaultForm();
      const el = buildContainer('row', []);
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });
      expect(getAllowedDropRegions(form, getElement(form, form.rootElementId))).toEqual([]);
    });

    it("should return ['left', 'right'] when the target's parent has a row layout", () => {
      const form = getDefaultForm();
      const parent = buildContainer('row', []);
      addElement({
        form,
        element: parent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildContainer('row', []);
      addElement({
        form,
        element: el,
        parentId: parent.id,
        index: 0,
      });
      const child = buildText('foo');
      addElement({
        form,
        element: child,
        parentId: el.id,
        index: 0,
      });
      expect(getAllowedDropRegions(form, el)).toEqual(['left', 'right']);
    });

    it("should return ['top', 'bottom'] when the target's parent has a column layout", () => {
      const form = getDefaultForm();
      const parent = buildContainer('column', []);
      addElement({
        form,
        element: parent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildContainer('column', []);
      addElement({
        form,
        element: el,
        parentId: parent.id,
        index: 0,
      });
      const child = buildText('foo');
      addElement({
        form,
        element: child,
        parentId: el.id,
        index: 0,
      });
      expect(getAllowedDropRegions(form, el)).toEqual(['top', 'bottom']);
    });
  });

  describe('reparentElement', () => {
    it('should move the element to the new parent (removing it from the old parent)', () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const newParent = buildContainer('row', []);
      addElement({
        form,
        element: newParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });

      reparentElement({
        form,
        id: el.id,
        newParentId: newParent.id,
        index: 0,
      });

      expect(getElement(form, el.id).parentId).toBe(newParent.id);
      expect(getElement(form, newParent.id, isContainerElement).data.children.length).toBe(1);
      expect(getElement(form, newParent.id, isContainerElement).data.children[0]).toBe(el.id);
      expect(getElement(form, oldParent.id, isContainerElement).data.children).toEqual([]);
    });
    it('should insert the element at the specified index in the new parent', () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const newParent = buildContainer('row', []);
      addElement({
        form,
        element: newParent,
        parentId: form.rootElementId,
        index: 0,
      });
      addElement({
        form,
        element: buildText('bar'),
        parentId: newParent.id,
        index: 0,
      });
      addElement({
        form,
        element: buildText('baz'),
        parentId: newParent.id,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });

      reparentElement({
        form,
        id: el.id,
        newParentId: newParent.id,
        index: 1,
      });

      expect(getElement(form, el.id).parentId).toBe(newParent.id);
      expect(getElement(form, newParent.id, isContainerElement).data.children.length).toBe(3);
      expect(getElement(form, newParent.id, isContainerElement).data.children[1]).toBe(el.id);
      expect(getElement(form, oldParent.id, isContainerElement).data.children).toEqual([]);
    });

    it('should return true if the element was reparented', () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const newParent = buildContainer('row', []);
      addElement({
        form,
        element: newParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });

      expect(
        reparentElement({
          form,
          id: el.id,
          newParentId: newParent.id,
          index: 0,
        })
      ).toBe(true);
    });

    it('should noop and return true if the new parent is the old parent', () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });

      const prevForm = deepClone(form);

      expect(
        reparentElement({
          form,
          id: el.id,
          newParentId: oldParent.id,
          index: 0,
        })
      ).toBe(true);

      expect(form).toEqual(prevForm);
    });

    it('should noop and return false if the element does not exist', () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const newParent = buildContainer('row', []);
      addElement({
        form,
        element: newParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');

      const prevForm = deepClone(form);

      expect(
        reparentElement({
          form,
          id: el.id,
          newParentId: newParent.id,
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it('should noop and return false if the is the root', () => {
      const form = getDefaultForm();
      const rootElement = getElement(form, form.rootElementId);

      const prevForm = deepClone(form);

      expect(
        reparentElement({
          form,
          id: rootElement.id,
          newParentId: 'non-existent-id',
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it("should noop and return false if the old parent doesn't exist", () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });
      el.parentId = 'non-existent-id';

      const prevForm = deepClone(form);

      expect(
        reparentElement({
          form,
          id: el.id,
          newParentId: oldParent.id,
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it("should noop and return false if the old parent isn't a container", () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const notAContainer = buildText('foo');
      addElement({
        form,
        element: notAContainer,
        parentId: form.rootElementId,
        index: 0,
      });
      const newParent = buildContainer('row', []);
      addElement({
        form,
        element: newParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });
      el.parentId = notAContainer.id;

      const prevForm = deepClone(form);

      expect(
        reparentElement({
          form,
          id: el.id,
          newParentId: newParent.id,
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it("should noop and return false if the new parent doesn't exist", () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });

      const prevForm = deepClone(form);

      expect(
        reparentElement({
          form,
          id: el.id,
          newParentId: 'non-existent-id',
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });

    it("should noop and return false if the old parent isn't a container", () => {
      const form = getDefaultForm();
      const oldParent = buildContainer('row', []);
      addElement({
        form,
        element: oldParent,
        parentId: form.rootElementId,
        index: 0,
      });
      const notAContainer = buildText('foo');
      addElement({
        form,
        element: notAContainer,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: oldParent.id,
        index: 0,
      });

      const prevForm = deepClone(form);

      expect(
        reparentElement({
          form,
          id: el.id,
          newParentId: notAContainer.id,
          index: 0,
        })
      ).toBe(false);

      expect(form).toEqual(prevForm);
    });
  });

  describe('validateFormStructure', () => {
    it('should be happy with the default form', () => {
      const form = getDefaultForm();
      expect(validateFormStructure(form)).toBe(true);
    });

    it('should return true if all children are reachable and there are no extra elements', () => {
      const form = getDefaultForm();
      const container1 = buildContainer('row', []);
      addElement({
        form,
        element: container1,
        parentId: form.rootElementId,
        index: 0,
      });
      const container2 = buildContainer('row', []);
      addElement({
        form,
        element: container2,
        parentId: container1.id,
        index: 0,
      });
      const child = buildText('foo');
      addElement({
        form,
        element: child,
        parentId: container2.id,
        index: 0,
      });

      expect(validateFormStructure(form)).toBe(true);
    });

    it('should return false if a child is not reachable', () => {
      const form = getDefaultForm();
      const parent = buildContainer('row', []);
      addElement({
        form,
        element: parent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: parent.id,
        index: 0,
      });
      el.parentId = 'non-existent-parent';
      parent.data.children = ['non-existent-child'];

      expect(validateFormStructure(form)).toBe(false);
    });

    it("should return false if a non-root child's parent does not exist", () => {
      const form = getDefaultForm();
      const parent = buildContainer('row', []);
      addElement({
        form,
        element: parent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: parent.id,
        index: 0,
      });
      el.parentId = 'non-existent-parent';
      expect(validateFormStructure(form)).toBe(false);
    });

    it('should be OK with the root not having a parent', () => {
      // This test is the same as the default form
      const form = getDefaultForm();
      const rootElement = form.elements[form.rootElementId];
      assert(rootElement !== undefined);
      expect(rootElement.parentId).toBeUndefined();
      expect(validateFormStructure(form)).toBe(true);
    });

    it("should return false if a child's parent is not a container", () => {
      const form = getDefaultForm();
      const notAContainer = buildText('foo');
      addElement({
        form,
        element: notAContainer,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('bar');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });
      el.parentId = notAContainer.id;
      expect(validateFormStructure(form)).toBe(false);
    });

    it("should return false if the child's parent does not have the child in its children list", () => {
      const form = getDefaultForm();
      const parent = buildContainer('row', []);
      addElement({
        form,
        element: parent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: parent.id,
        index: 0,
      });
      parent.data.children = [];
      expect(validateFormStructure(form)).toBe(false);
    });

    it('should return false if there are extra elements', () => {
      const form = getDefaultForm();
      const parent = buildContainer('row', []);
      addElement({
        form,
        element: parent,
        parentId: form.rootElementId,
        index: 0,
      });
      const el = buildText('foo');
      el.parentId = parent.id;
      form.elements[el.id] = el;
      expect(validateFormStructure(form)).toBe(false);
    });

    it('should return false if the root element is not a container', () => {
      const el = buildText('foo');
      const form: BuilderForm = {
        elements: {
          [el.id]: el,
        },
        rootElementId: el.id,
      };
      expect(validateFormStructure(form)).toBe(false);
    });
  });

  describe('getIsFormEmpty', () => {
    it('should return true if the form is empty', () => {
      const form = getDefaultForm();
      expect(getIsFormEmpty(form)).toBe(true);
    });

    it('should return false if the form is not empty', () => {
      const form = getDefaultForm();
      const el = buildText('foo');
      addElement({
        form,
        element: el,
        parentId: form.rootElementId,
        index: 0,
      });
      expect(getIsFormEmpty(form)).toBe(false);
    });
  });
});
