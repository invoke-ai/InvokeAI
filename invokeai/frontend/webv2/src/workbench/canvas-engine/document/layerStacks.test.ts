import type { CanvasStackForests } from '@workbench/canvas-engine/contracts';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { getDocumentIndex } from './documentIndex';
import {
  getSiblingOrder,
  haveSameStructure,
  LAYER_STACK_ORDER,
  LAYER_STACKS_TOP_FIRST,
  moveNodesWithinSiblings,
  reorderSiblings,
} from './layerStacks';

const layer = layerContract;
const group = groupContract;

/** raster: r1, g1[r2, r3], r4 · control: c1 */
const stacks = (): CanvasStackForests =>
  stacksFrom([layer('r1'), group('g1', [layer('r2'), layer('r3')]), layer('r4'), layer('c1', 'control')]);

const ids = (forest: CanvasStackForests | null): string[] | undefined =>
  forest ? getDocumentIndex({ stacks: forest }).nodes.map((entry) => entry.node.id) : undefined;

describe('layer stacks', () => {
  it('declares one composite order and its panel mirror', () => {
    expect(LAYER_STACK_ORDER).toEqual(['raster', 'control', 'regional_guidance', 'inpaint_mask']);
    expect(LAYER_STACKS_TOP_FIRST).toEqual([...LAYER_STACK_ORDER].reverse());
  });

  it('reads a sibling list for the root and for a group, empty lists included', () => {
    expect(getSiblingOrder(stacks(), 'raster', null)).toEqual({
      orderedIds: ['r1', 'g1', 'r4'],
      parentId: null,
      stack: 'raster',
    });
    expect(getSiblingOrder(stacks(), 'raster', 'g1')).toEqual({
      orderedIds: ['r2', 'r3'],
      parentId: 'g1',
      stack: 'raster',
    });
    expect(getSiblingOrder(stacks(), 'regional_guidance', null)).toEqual({
      orderedIds: [],
      parentId: null,
      stack: 'regional_guidance',
    });
    expect(getSiblingOrder(stacks(), 'raster', 'r1')).toEqual({ orderedIds: [], parentId: 'r1', stack: 'raster' });
  });

  it('reorders one sibling list and shares every other node', () => {
    const before = stacks();
    const next = reorderSiblings(before, { orderedIds: ['r3', 'r2'], parentId: 'g1', stack: 'raster' });

    expect(ids(next)).toEqual(['c1', 'r1', 'g1', 'r3', 'r2', 'r4']);
    expect(next?.control).toBe(before.control);
    expect(next?.raster[0]).toBe(before.raster[0]);
    expect(next?.raster[2]).toBe(before.raster[2]);
  });

  it('compares forests by structure, not identity', () => {
    expect(haveSameStructure(stacks(), stacks())).toBe(true);
    expect(
      haveSameStructure(
        stacks(),
        stacksFrom([layer('r1'), group('g1', [layer('r3'), layer('r2')]), layer('r4'), layer('c1', 'control')])
      )
    ).toBe(false);
    expect(
      haveSameStructure(
        stacks(),
        stacksFrom([layer('r1'), layer('r2'), group('g1', [layer('r3')]), layer('r4'), layer('c1', 'control')])
      )
    ).toBe(false);
    expect(
      haveSameStructure(stacks(), stacksFrom([layer('r1'), group('g1', [layer('r2'), layer('r3')]), layer('r4')]))
    ).toBe(false);
  });

  it('accepts an unchanged order, a single member, and an empty list', () => {
    expect(ids(reorderSiblings(stacks(), { orderedIds: ['r1', 'g1', 'r4'], parentId: null, stack: 'raster' }))).toEqual(
      ids(stacks())
    );
    expect(ids(reorderSiblings(stacks(), { orderedIds: ['c1'], parentId: null, stack: 'control' }))).toEqual(
      ids(stacks())
    );
    expect(ids(reorderSiblings(stacks(), { orderedIds: [], parentId: null, stack: 'regional_guidance' }))).toEqual(
      ids(stacks())
    );
  });

  it.each([
    ['a missing member', 'raster', null, ['r1']],
    ['a duplicate', 'raster', null, ['r1', 'r1', 'r4']],
    ['a foreign id', 'raster', null, ['r1', 'g1', 'x']],
    ['a member of another list', 'raster', null, ['r1', 'g1', 'r2']],
    ['a leaf as parent', 'raster', 'r1', ['r2']],
    ['members for an empty list', 'regional_guidance', null, ['r1']],
  ] as const)('refuses %s', (_label, stack, parentId, orderedIds) => {
    expect(reorderSiblings(stacks(), { orderedIds, parentId, stack })).toBeNull();
  });
});

describe('moveNodesWithinSiblings', () => {
  /** raster: r1, g1[r2, r3, r4], r5 · inpaint: i1, i2 */
  const forest = (): CanvasStackForests =>
    stacksFrom([
      layer('r1'),
      group('g1', [layer('r2'), layer('r3'), layer('r4')]),
      layer('r5'),
      layer('i1', 'inpaint_mask'),
      layer('i2', 'inpaint_mask'),
    ]);
  const index = () => getDocumentIndex({ stacks: forest() });

  it('moves selected nodes one place within each sibling list, lists ordered by their first selected member', () => {
    expect(moveNodesWithinSiblings(index(), ['i2', 'r3', 'r5'], 'forward')).toEqual([
      { orderedIds: ['i2', 'i1'], parentId: null, stack: 'inpaint_mask' },
      { orderedIds: ['r3', 'r2', 'r4'], parentId: 'g1', stack: 'raster' },
      { orderedIds: ['r1', 'r5', 'g1'], parentId: null, stack: 'raster' },
    ]);
  });

  it('moves a selection to the back of its own list while preserving its order', () => {
    expect(moveNodesWithinSiblings(index(), ['r2', 'r3'], 'back')).toEqual([
      { orderedIds: ['r4', 'r2', 'r3'], parentId: 'g1', stack: 'raster' },
    ]);
  });

  it('moves a selected group with its descendants and ignores selected descendants of a selected ancestor', () => {
    expect(moveNodesWithinSiblings(index(), ['g1', 'r3'], 'front')).toEqual([
      { orderedIds: ['g1', 'r1', 'r5'], parentId: null, stack: 'raster' },
    ]);
  });

  it('returns no commands when every selected node is already at the requested boundary', () => {
    expect(moveNodesWithinSiblings(index(), ['r1', 'r2', 'i1'], 'front')).toEqual([]);
    expect(moveNodesWithinSiblings(index(), ['ghost'], 'front')).toEqual([]);
  });
});
