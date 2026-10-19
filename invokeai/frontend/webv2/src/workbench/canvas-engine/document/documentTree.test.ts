import type { CanvasGroupContract, CanvasStackForests } from '@workbench/canvas-engine/contracts';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import {
  getDocumentIndex,
  getValueEditVisitCount,
  isSelfOrAncestor,
  outermostNodes,
  resetDocumentIndexBuildCount,
  updateNodeValues,
  valueEditBetween,
} from './documentIndex';
import {
  cloneSubtree,
  collectSubtree,
  collectSubtreeLeaves,
  insertNodes,
  removeNodes,
  subtreeDepth,
} from './documentTree';

const layer = layerContract;
const group = groupContract;

/** raster: r1, g1[r2, g2[r3]], r4 · control: c1 */
const stacks = (): CanvasStackForests =>
  stacksFrom([
    layer('r1'),
    group('g1', [layer('r2'), group('g2', [layer('r3')])]),
    layer('r4'),
    layer('c1', 'control'),
  ]);

const g1 = (forest: CanvasStackForests): CanvasGroupContract => forest.raster[1] as CanvasGroupContract;

describe('document tree', () => {
  it('walks subtrees in preorder and measures nesting', () => {
    const forest = stacks();
    expect(collectSubtree(g1(forest)).map((node) => node.id)).toEqual(['g1', 'r2', 'g2', 'r3']);
    expect(collectSubtreeLeaves(g1(forest)).map((node) => node.id)).toEqual(['r2', 'r3']);
    expect(subtreeDepth(g1(forest))).toBe(2);
    expect(subtreeDepth(forest.raster[0]!)).toBe(0);
    expect(subtreeDepth(group('empty'))).toBe(0);
  });

  it('updates nodes in place while sharing every untouched node, and returns the same forests for a no-op', () => {
    const before = stacks();
    getDocumentIndex({ stacks: before });
    resetDocumentIndexBuildCount();
    const after = updateNodeValues(before, new Map([['r3', (node) => ({ ...node, name: 'renamed' })]]));
    // r3 and its two ancestors; r1, r2, r4 and c1 are never visited.
    expect(getValueEditVisitCount()).toBe(3);

    expect(after).not.toBe(before);
    expect(after.control).toBe(before.control);
    expect(after.raster[0]).toBe(before.raster[0]);
    expect(after.raster[2]).toBe(before.raster[2]);
    expect(g1(after)).not.toBe(g1(before));
    expect(g1(after).children[0]).toBe(g1(before).children[0]);
    expect(collectSubtreeLeaves(g1(after))[1]?.name).toBe('renamed');
    expect(updateNodeValues(before, new Map([['r3', (node) => node]]))).toBe(before);
    expect(updateNodeValues(before, new Map([['ghost', (node) => ({ ...node, name: 'x' })]]))).toBe(before);
  });

  it('records the replaced nodes of each value edit and folds consecutive edits for a consumer that fell behind', () => {
    const before = stacks();
    const renamed = updateNodeValues(
      before,
      new Map([
        ['g2', (node) => ({ ...node, isEnabled: false })],
        ['r3', (node) => ({ ...node, name: 'renamed' })],
      ])
    );
    const edit = valueEditBetween(before, renamed)!;
    // Only the updated nodes are listed: g1 was rebuilt for a new child array, not edited.
    expect([...edit.keys()]).toEqual(['r3', 'g2']);
    expect(edit.get('r3')!.before).toBe(collectSubtreeLeaves(g1(before))[1]);
    expect((edit.get('g2')!.after as CanvasGroupContract).children[0]).toBe(edit.get('r3')!.after);

    const moved = updateNodeValues(renamed, new Map([['r3', (node) => ({ ...node, opacity: 0.5 })]]));
    const folded = valueEditBetween(before, moved)!;
    expect(folded.get('r3')!.before).toBe(edit.get('r3')!.before);
    expect(folded.get('r3')!.after).toMatchObject({ name: 'renamed', opacity: 0.5 });
    expect(folded.has('g2')).toBe(true);
    // Structural edits are not value edits: nothing connects the forests across a removal.
    expect(valueEditBetween(before, removeNodes(moved, new Set(['r4'])))).toBeNull();
  });

  it('stops folding once the folded edits would touch more than a quarter of a large forest', () => {
    const before = stacksFrom(Array.from({ length: 400 }, (_, index) => layer(`l${index}`)));
    const fade = (stacks: CanvasStackForests, ids: readonly string[], opacity: number) =>
      updateNodeValues(stacks, new Map(ids.map((id) => [id, (node) => ({ ...node, opacity })])));
    const first = Array.from({ length: 40 }, (_, index) => `l${index}`);
    const once = fade(before, first, 0.5);

    // 80 replaced nodes across two steps stay within a quarter of 400; 110 do not.
    expect(valueEditBetween(before, fade(once, first, 0.4))?.size).toBe(40);
    const wide = Array.from({ length: 70 }, (_, index) => `l${index + 100}`);
    expect(valueEditBetween(before, fade(once, wide, 0.4))).toBeNull();
    expect(valueEditBetween(once, fade(once, wide, 0.4))?.size).toBe(70);
  });

  it('removes subtrees and shares the untouched forests', () => {
    const before = stacks();
    const after = removeNodes(before, new Set(['g2', 'r4']));

    expect(getDocumentIndex({ stacks: after }).nodes.map((entry) => entry.node.id)).toEqual(['c1', 'r1', 'g1', 'r2']);
    expect(after.control).toBe(before.control);
    expect(removeNodes(before, new Set(['ghost']))).toBe(before);
  });

  it('inserts into the root or a group at a clamped index and refuses a leaf parent', () => {
    const before = stacks();
    expect(
      getDocumentIndex({ stacks: insertNodes(before, 'raster', null, 99, [layer('n')])! }).stacks.raster.at(-1)?.id
    ).toBe('n');
    expect(g1(insertNodes(before, 'raster', 'g1', 1, [layer('n')])!).children.map((node) => node.id)).toEqual([
      'r2',
      'n',
      'g2',
    ]);
    expect(insertNodes(before, 'raster', 'r1', 0, [layer('n')])).toBeNull();
  });

  it('clones a subtree with fresh ids and no shared objects', () => {
    let next = 0;
    const source = g1(stacks());
    const { ids, node } = cloneSubtree(source, () => `id${next++}`);

    expect(collectSubtree(node).map((entry) => entry.id)).toEqual(['id0', 'id1', 'id2', 'id3']);
    expect(ids.get('r3')).toBe('id3');
    expect((node as CanvasGroupContract).children[0]).not.toBe(source.children[0]);
    expect(collectSubtreeLeaves(node)[0]!.transform).not.toBe(collectSubtreeLeaves(source)[0]!.transform);
  });
});

describe('document index', () => {
  it('records parent, path, sibling index, order and ancestor-effective state per node', () => {
    const forest = stacksFrom([
      layer('r1'),
      group('g1', [layer('r2'), group('g2', [layer('r3')], { isLocked: true })], { isEnabled: false }),
      layer('c1', 'control', { isHidden: true } as never),
    ]);
    const index = getDocumentIndex({ stacks: forest });

    expect(index.nodes.map((entry) => entry.node.id)).toEqual(['c1', 'r1', 'g1', 'r2', 'g2', 'r3']);
    expect(index.leaves.map((leaf) => leaf.id)).toEqual(['c1', 'r1', 'r2', 'r3']);
    expect(index.byId.get('r3')).toMatchObject({
      ancestorsEnabled: false,
      ancestorsHidden: false,
      ancestorsLocked: true,
      order: 5,
      parentId: 'g2',
      path: ['g1', 'g2'],
      siblingIndex: 0,
      stack: 'raster',
    });
    expect(index.byId.get('r2')).toMatchObject({ ancestorsEnabled: false, ancestorsLocked: false, siblingIndex: 0 });
    expect(index.byId.get('c1')).toMatchObject({ ancestorsHidden: false, path: [], stack: 'control' });
    expect(index.maxDepth).toBe(2);
    expect(getDocumentIndex({ stacks: forest })).toBe(index);
  });

  it('answers ancestry and keeps only the outermost of a nested selection', () => {
    const index = getDocumentIndex({ stacks: stacks() });
    expect(isSelfOrAncestor(index, 'r3', 'g1')).toBe(true);
    expect(isSelfOrAncestor(index, 'g1', 'g1')).toBe(true);
    expect(isSelfOrAncestor(index, 'r1', 'g1')).toBe(false);
    expect(outermostNodes(index, ['r3', 'g1', 'r4', 'r2']).map((entry) => entry.node.id)).toEqual(['g1', 'r4']);
  });
});
