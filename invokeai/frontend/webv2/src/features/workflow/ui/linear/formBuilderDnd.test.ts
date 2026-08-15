import type { WorkflowForm } from '@features/workflow/contracts';

import { describe, expect, it } from 'vitest';

import { findFormParentId, isFormDescendantOrSelf, resolveFormDrop } from './formBuilderDnd';

/**
 * root -> [fieldA, container1 -> [fieldB], fieldC]
 * Mirrors the shape `getFormChildren` walks: a root container element in
 * `form.elements` whose `data.children` list the top-level elements, each
 * carrying its own `parentId` back to its container (exactly what
 * `moveFormElementTo` maintains in `core/document.ts`).
 */
const buildFormFixture = (): WorkflowForm => ({
  elements: {
    container1: {
      data: { children: ['fieldB'], layout: 'column' },
      id: 'container1',
      parentId: 'root',
      type: 'container',
    },
    fieldA: { data: { content: '' }, id: 'fieldA', parentId: 'root', type: 'heading' },
    fieldB: { data: { content: '' }, id: 'fieldB', parentId: 'container1', type: 'heading' },
    fieldC: { data: { content: '' }, id: 'fieldC', parentId: 'root', type: 'heading' },
    root: { data: { children: ['fieldA', 'container1', 'fieldC'], layout: 'column' }, id: 'root', type: 'container' },
  },
  rootElementId: 'root',
});

const form = buildFormFixture();

describe('resolveFormDrop', () => {
  it('drops above a root sibling at its index', () => {
    expect(resolveFormDrop(form, 'fieldC', { edge: 'above', elementId: 'fieldA', kind: 'edge' })).toEqual({
      index: 0,
      parentId: form.rootElementId,
    });
  });

  it('drops below a nested element inside its container', () => {
    expect(resolveFormDrop(form, 'fieldA', { edge: 'below', elementId: 'fieldB', kind: 'edge' })).toEqual({
      index: 1,
      parentId: 'container1',
    });
  });

  it('appends into a container body', () => {
    expect(resolveFormDrop(form, 'fieldA', { containerId: 'container1', kind: 'into' })).toEqual({
      index: 1,
      parentId: 'container1',
    });
  });

  it('rejects dropping a container into itself or a descendant', () => {
    expect(resolveFormDrop(form, 'container1', { containerId: 'container1', kind: 'into' })).toBeNull();
    expect(resolveFormDrop(form, 'container1', { edge: 'above', elementId: 'fieldB', kind: 'edge' })).toBeNull();
  });

  it('rejects dropping an element onto itself', () => {
    expect(resolveFormDrop(form, 'fieldA', { edge: 'above', elementId: 'fieldA', kind: 'edge' })).toBeNull();
  });
});

describe('isFormDescendantOrSelf', () => {
  it('sees a nested child as a descendant', () => {
    expect(isFormDescendantOrSelf(form, 'container1', 'fieldB')).toBe(true);
    expect(isFormDescendantOrSelf(form, 'container1', 'fieldA')).toBe(false);
  });
});

describe('findFormParentId', () => {
  it('finds root and container parents', () => {
    expect(findFormParentId(form, 'fieldA')).toBe(form.rootElementId);
    expect(findFormParentId(form, 'fieldB')).toBe('container1');
  });

  it('returns null for the root element itself', () => {
    expect(findFormParentId(form, form.rootElementId)).toBeNull();
  });
});
