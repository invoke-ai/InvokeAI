import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { projectLayerDrop } from './layerTreeDrop';
import { buildLayerStackRows } from './layerTreeRows';

const layer = layerContract;
const group = groupContract;

/** raster: r1, G[r2, H[r3], r4], r5 · control: c1 */
const document = () =>
  documentFrom([
    layer('c1', 'control'),
    layer('r1'),
    group('G', [layer('r2'), group('H', [layer('r3')], { isLocked: true }), layer('r4')], { isEnabled: false }),
    layer('r5'),
  ]);

const outline = (rows: readonly { id: string; vm: { depth: number } }[]) =>
  rows.map((row) => `${'  '.repeat(row.vm.depth)}${row.id}`);

describe('buildLayerStackRows', () => {
  it('lists rendered rows per stack, hiding the children of collapsed groups', () => {
    const collapsed = buildLayerStackRows(document().stacks, new Set());
    expect(outline(collapsed.raster.rows)).toEqual(['r1', 'G', 'r5']);
    expect(collapsed.raster.nodeIds).toEqual(['r1', 'G', 'r2', 'H', 'r3', 'r4', 'r5']);
    expect(collapsed.raster.leafCount).toBe(5);
    expect(outline(collapsed.control.rows)).toEqual(['c1']);
    expect(collapsed.inpaint_mask.rows).toEqual([]);

    const expanded = buildLayerStackRows(document().stacks, new Set(['G', 'H']));
    expect(outline(expanded.raster.rows)).toEqual(['r1', 'G', '  r2', '  H', '    r3', '  r4', 'r5']);
    expect(outline(buildLayerStackRows(document().stacks, new Set(['H'])).raster.rows)).toEqual(['r1', 'G', 'r5']);
  });

  it('carries effective state and counts on each row', () => {
    const rows = buildLayerStackRows(document().stacks, new Set(['G', 'H'])).raster.rows;
    const byId = Object.fromEntries(rows.map((row) => [row.id, row]));
    expect(byId.G).toMatchObject({
      expanded: true,
      posInSet: 2,
      setSize: 3,
      vm: { childCount: 3, contributionEnabled: false, kind: 'group', leafCount: 3 },
    });
    expect(byId.r3).toMatchObject({
      posInSet: 1,
      setSize: 1,
      vm: {
        ancestorsEnabled: false,
        contributionEnabled: false,
        depth: 2,
        effectiveLocked: true,
        kind: 'leaf',
        parentId: 'H',
      },
    });
    expect(byId.H).toMatchObject({ vm: { ancestorsLocked: false, contributionEnabled: false, effectiveLocked: true } });
    expect(byId.r5).toMatchObject({ posInSet: 3, vm: { contributionEnabled: true, effectiveLocked: false } });
  });

  it('keeps row identity for untouched nodes across an unrelated edit and a toggle elsewhere', () => {
    const before = buildLayerStackRows(document().stacks, new Set(['G'])).raster.rows;
    const renamed = documentFrom([
      layer('c1', 'control'),
      layer('r1', 'raster', { name: 'renamed' }),
      before[1]!.vm.node,
      layer('r5'),
    ]);
    const after = buildLayerStackRows(renamed.stacks, new Set(['G'])).raster.rows;
    expect(after[1]).toBe(before[1]);
    expect(after[2]).toBe(before[2]);
    expect(after[0]).not.toBe(before[0]);
    const toggled = buildLayerStackRows(renamed.stacks, new Set(['G', 'H'])).raster.rows;
    expect(toggled.find((row) => row.id === 'G')).toBe(after[1]);
    expect(toggled.find((row) => row.id === 'H')).not.toBe(after.find((row) => row.id === 'H'));
  });
});

describe('projectLayerDrop', () => {
  const rows = () => buildLayerStackRows(document().stacks, new Set(['G', 'H'])).raster.rows;

  it('nests into a hovered group on the inside edge, including an empty one', () => {
    // Into G, at its top, above its first rendered child.
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 0, edge: 'inside', overId: 'G', rows: rows() })
    ).toMatchObject({
      beforeId: 'r2',
      depth: 1,
      ids: ['r5'],
      parentId: 'G',
      stack: 'raster',
    });
    // An empty expanded group has no child rows; the inside edge is the way in.
    const withEmpty = documentFrom([layer('r1'), group('E', []), layer('r5')]);
    const emptyRows = buildLayerStackRows(withEmpty.stacks, new Set(['E'])).raster.rows;
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 0, edge: 'inside', overId: 'E', rows: emptyRows })
    ).toMatchObject({
      beforeId: null,
      depth: 1,
      ids: ['r5'],
      parentId: 'E',
      stack: 'raster',
    });
    // A COLLAPSED group with children drops at its top via the model child.
    const collapsedRows = buildLayerStackRows(document().stacks, new Set()).raster.rows;
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 0, edge: 'inside', overId: 'G', rows: collapsedRows })
    ).toMatchObject({
      beforeId: 'r2',
      depth: 1,
      ids: ['r5'],
      parentId: 'G',
      stack: 'raster',
    });
    // The inside edge means nothing on a leaf.
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 0, edge: 'inside', overId: 'r1', rows: rows() })
    ).toBeNull();
  });

  it('drops between siblings at the same depth', () => {
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 0, edge: 'above', overId: 'r1', rows: rows() })
    ).toMatchObject({
      beforeId: 'r1',
      depth: 0,
      ids: ['r5'],
      parentId: null,
      stack: 'raster',
    });
    expect(
      projectLayerDrop({ activeIds: ['r1'], depthOffset: 0, edge: 'below', overId: 'r5', rows: rows() })
    ).toMatchObject({
      beforeId: null,
      depth: 0,
      ids: ['r1'],
      parentId: null,
      stack: 'raster',
    });
  });

  it('indents into the group above when dragged right, and outdents at a subtree end when dragged left', () => {
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 1, edge: 'below', overId: 'G', rows: rows() })
    ).toMatchObject({
      beforeId: 'r2',
      depth: 1,
      ids: ['r5'],
      parentId: 'G',
      stack: 'raster',
    });
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 2, edge: 'below', overId: 'H', rows: rows() })
    ).toMatchObject({
      beforeId: 'r3',
      depth: 2,
      ids: ['r5'],
      parentId: 'H',
      stack: 'raster',
    });
    expect(
      projectLayerDrop({ activeIds: ['r1'], depthOffset: 0, edge: 'below', overId: 'r4', rows: rows() })
    ).toMatchObject({
      beforeId: 'r5',
      depth: 0,
      ids: ['r1'],
      parentId: null,
      stack: 'raster',
    });
    expect(
      projectLayerDrop({ activeIds: ['r1'], depthOffset: 1, edge: 'below', overId: 'r4', rows: rows() })
    ).toMatchObject({
      beforeId: null,
      depth: 1,
      ids: ['r1'],
      parentId: 'G',
      stack: 'raster',
    });
  });

  it('clamps the depth to what the neighbours allow', () => {
    // Between r2 and H the only legal depth is 1.
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 5, edge: 'below', overId: 'r2', rows: rows() })
    ).toMatchObject({ depth: 1, parentId: 'G' });
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: -5, edge: 'below', overId: 'r2', rows: rows() })
    ).toMatchObject({ depth: 1, parentId: 'G' });
  });

  it('moves a group with its descendants and refuses to drop into itself', () => {
    expect(
      projectLayerDrop({ activeIds: ['G', 'r3'], depthOffset: 0, edge: 'below', overId: 'r5', rows: rows() })
    ).toMatchObject({
      beforeId: null,
      depth: 0,
      ids: ['G'],
      parentId: null,
      stack: 'raster',
    });
    expect(
      projectLayerDrop({ activeIds: ['G'], depthOffset: 0, edge: 'above', overId: 'r3', rows: rows() })
    ).toBeNull();
  });

  it('keeps a multi-selection in document order', () => {
    expect(
      projectLayerDrop({ activeIds: ['r5', 'r1'], depthOffset: 0, edge: 'above', overId: 'r4', rows: rows() })
    ).toMatchObject({
      beforeId: 'r4',
      depth: 1,
      ids: ['r1', 'r5'],
      parentId: 'G',
      stack: 'raster',
    });
  });

  it('treats a collapsed group as a leaf for indentation and refuses past the depth limit', () => {
    const collapsed = buildLayerStackRows(document().stacks, new Set()).raster.rows;
    expect(
      projectLayerDrop({ activeIds: ['r5'], depthOffset: 1, edge: 'below', overId: 'G', rows: collapsed })
    ).toMatchObject({ depth: 0, parentId: null });
    const nest = (depth: number): ReturnType<typeof layer> | ReturnType<typeof group> =>
      depth === 0 ? layer('leaf') : group(`n${depth}`, [nest(depth - 1)]);
    const deep = documentFrom([nest(10), group('S', [layer('s')])]);
    const ids = new Set(Array.from({ length: 10 }, (_, index) => `n${index + 1}`).concat(['S']));
    const deepRows = buildLayerStackRows(deep.stacks, ids).raster.rows;
    expect(
      projectLayerDrop({ activeIds: ['S'], depthOffset: 10, edge: 'below', overId: 'leaf', rows: deepRows })
    ).toMatchObject({ depth: 9, parentId: 'n2' });
    expect(
      projectLayerDrop({ activeIds: ['s'], depthOffset: 10, edge: 'below', overId: 'leaf', rows: deepRows })
    ).toMatchObject({ depth: 10, parentId: 'n1' });
    expect(
      projectLayerDrop({ activeIds: ['S'], depthOffset: 10, edge: 'below', overId: 'n1', rows: deepRows })
    ).toBeNull();
  });
});

describe('buildLayerStackRows filter', () => {
  it('keeps matches and their ancestors, opens only groups with a match beneath, and leaves a matching group closed', () => {
    const filtered = buildLayerStackRows(document().stacks, new Set(), 'r3');
    expect(outline(filtered.raster.rows)).toEqual(['G', '  H', '    r3']);
    expect(filtered.raster.rows.map((row) => row.expanded)).toEqual([true, true, false]);
    expect(filtered.control.rows).toEqual([]);
    const groupMatch = buildLayerStackRows(document().stacks, new Set(['G']), 'g');
    expect(outline(groupMatch.raster.rows)).toEqual(['G']);
    expect(groupMatch.raster.rows[0]!.expanded).toBe(false);
  });
});
