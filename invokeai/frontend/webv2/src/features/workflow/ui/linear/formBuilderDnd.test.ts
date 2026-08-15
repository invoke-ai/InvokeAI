import type { WorkflowForm } from '@features/workflow/contracts';

import { describe, expect, it } from 'vitest';

import {
  findFormParentId,
  formEdgeDroppableId,
  formIntoDroppableId,
  getFormDropEdge,
  isFormDescendantOrSelf,
  pickInnermostFormCollision,
  resolveFormDrop,
} from './formBuilderDnd';

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

describe('getFormDropEdge', () => {
  it('splits at the midline of the hovered card', () => {
    expect(getFormDropEdge(10, 0, 40)).toBe('above');
    expect(getFormDropEdge(30, 0, 40)).toBe('below');
  });
});

/**
 * root -> [container A -> [container B -> [fieldD], fieldC]]
 * Mirrors a nested-container layout the way the builder renders it: A's own
 * body holds both B (another container) and a plain field.
 */
const buildNestedFormFixture = (): WorkflowForm => ({
  elements: {
    A: { data: { children: ['B', 'fieldC'], layout: 'column' }, id: 'A', parentId: 'root', type: 'container' },
    B: { data: { children: ['fieldD'], layout: 'column' }, id: 'B', parentId: 'A', type: 'container' },
    fieldC: { data: { content: '' }, id: 'fieldC', parentId: 'A', type: 'heading' },
    fieldD: { data: { content: '' }, id: 'fieldD', parentId: 'B', type: 'heading' },
    root: { data: { children: ['A'], layout: 'column' }, id: 'root', type: 'container' },
  },
  rootElementId: 'root',
});

describe('pickInnermostFormCollision', () => {
  const nestedForm = buildNestedFormFixture();

  it('returns null when there are no collisions', () => {
    expect(pickInnermostFormCollision([], nestedForm)).toBeNull();
  });

  it('prefers the deepest container when nested droppables overlap', () => {
    const collisions = [{ id: formIntoDroppableId('A') }, { id: formIntoDroppableId('B') }];

    expect(pickInnermostFormCollision(collisions, nestedForm)).toBe(formIntoDroppableId('B'));
  });

  it('is order-independent for nested containers', () => {
    const collisions = [{ id: formIntoDroppableId('B') }, { id: formIntoDroppableId('A') }];

    expect(pickInnermostFormCollision(collisions, nestedForm)).toBe(formIntoDroppableId('B'));
  });

  it('prefers an edge target over its enclosing container', () => {
    const collisions = [{ id: formIntoDroppableId('A') }, { id: formEdgeDroppableId('fieldC') }];

    expect(pickInnermostFormCollision(collisions, nestedForm)).toBe(formEdgeDroppableId('fieldC'));
  });

  /**
   * A container's own `into:X` dropzone is DOM-nested inside its own
   * `edge:X` box (the whole card), so hovering the dropzone always matches
   * both — not just when a *different*, deeper element's edge is involved.
   * `into:X` (lands inside X) must win over `edge:X` (lands inside X's
   * parent): dropping onto a container's own empty-state hint has to append
   * into it, not reorder it as a sibling.
   */
  it('prefers a container over its own edge when both match the same element', () => {
    const collisions = [{ id: formEdgeDroppableId('A') }, { id: formIntoDroppableId('A') }];

    expect(pickInnermostFormCollision(collisions, nestedForm)).toBe(formIntoDroppableId('A'));
    expect(pickInnermostFormCollision([...collisions].reverse(), nestedForm)).toBe(formIntoDroppableId('A'));
  });

  /**
   * root -> [C, B -> [D]]. C and B are incomparable siblings (neither is an
   * ancestor of the other); D is nested under B. A greedy "compare each new
   * candidate only against the current winner" pass can get stuck on C: C
   * isn't D's ancestor, so a pairwise check never promotes D once C already
   * won. Depth is computed independently per candidate specifically to avoid
   * this, so the deepest (`D`) must win regardless of arrival order.
   */
  it('is order-independent for incomparable siblings plus a nested descendant', () => {
    const siblingForm: WorkflowForm = {
      elements: {
        B: { data: { children: ['D'], layout: 'column' }, id: 'B', parentId: 'root', type: 'container' },
        C: { data: { children: [], layout: 'column' }, id: 'C', parentId: 'root', type: 'container' },
        D: { data: { children: [], layout: 'column' }, id: 'D', parentId: 'B', type: 'container' },
        root: { data: { children: ['C', 'B'], layout: 'column' }, id: 'root', type: 'container' },
      },
      rootElementId: 'root',
    };
    const collisions = [
      { id: formIntoDroppableId('C') },
      { id: formIntoDroppableId('B') },
      { id: formIntoDroppableId('D') },
    ];

    expect(pickInnermostFormCollision(collisions, siblingForm)).toBe(formIntoDroppableId('D'));
    expect(pickInnermostFormCollision([...collisions].reverse(), siblingForm)).toBe(formIntoDroppableId('D'));
  });

  it('falls back to the first collision when nothing parses as a form droppable', () => {
    expect(pickInnermostFormCollision([{ id: 'not-a-form-id' }], nestedForm)).toBe('not-a-form-id');
  });
});
