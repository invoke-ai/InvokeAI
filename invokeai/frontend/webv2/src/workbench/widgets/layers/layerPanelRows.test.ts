import type { CanvasAdjustmentEntry, RegionalGuidanceReferenceImage } from '@workbench/canvas-engine/api';

import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import type { PanelRow } from './layerPanelRows';

import { projectLayerChildRows } from './layerChildRows';
import { flattenPanelRows, navigateTree, panelRowHeight } from './layerPanelRows';
import { buildLayerStackRows } from './layerTreeRows';

const referenceImage = (id: string): RegionalGuidanceReferenceImage => ({
  config: {
    beginEndStepPct: [0, 1],
    clipVisionModel: 'ViT-H',
    image: null,
    method: 'full',
    model: null,
    type: 'ip_adapter',
    weight: 1,
  },
  id,
  isEnabled: true,
});

/** regional: rg1[ref1, ref2], rg2 · raster: r1 */
const stacks = () =>
  buildLayerStackRows(
    documentFrom([
      layerContract('rg1', 'regional_guidance', { referenceImages: [referenceImage('ref1'), referenceImage('ref2')] }),
      layerContract('rg2', 'regional_guidance'),
      layerContract('r1'),
    ]).stacks,
    new Set()
  );

const open = () => false;

const flatten = (collapsedLayerIds: ReadonlySet<string> = new Set()) =>
  flattenPanelRows(stacks(), [], open, { collapsedLayerIds, rowsFor: (row) => projectLayerChildRows(row.vm) });

const keys = (rows: readonly PanelRow[]) => rows.map((row) => row.key);

describe('flattenPanelRows with child rows', () => {
  it('inserts child rows after their layer, honoring per-layer collapse', () => {
    expect(keys(flatten())).toEqual([
      'header:regional_guidance',
      'rg1',
      'child:rg1:ref1',
      'child:rg1:ref2',
      'rg2',
      'header:raster',
      'r1',
    ]);
    expect(keys(flatten(new Set(['rg1'])))).toEqual(['header:regional_guidance', 'rg1', 'rg2', 'header:raster', 'r1']);
  });

  it('marks owning rows with their child count and expansion, and leaves others bare', () => {
    const rows = flatten();
    const rg1 = rows.find((row) => row.key === 'rg1');
    const rg2 = rows.find((row) => row.key === 'rg2');
    expect(rg1).toMatchObject({ childCount: 2, childrenExpanded: true, kind: 'node' });
    expect(rg2).toMatchObject({ childCount: 0, childrenExpanded: false });
    expect(flatten(new Set(['rg1'])).find((row) => row.key === 'rg1')).toMatchObject({
      childCount: 2,
      childrenExpanded: false,
    });
  });

  it('projects nothing without a child-rows source and sizes child rows below layer rows', () => {
    expect(keys(flattenPanelRows(stacks(), [], open))).toEqual([
      'header:regional_guidance',
      'rg1',
      'rg2',
      'header:raster',
      'r1',
    ]);
    const child = flatten().find((row) => row.kind === 'child')!;
    const node = flatten().find((row) => row.kind === 'node')!;
    expect(panelRowHeight(child)).toBeLessThan(panelRowHeight(node));
  });
});

describe('navigateTree over child rows', () => {
  it('walks child rows vertically like any other row', () => {
    const rows = flatten();
    expect(navigateTree(rows, 'rg1', 'ArrowDown')).toEqual({ focus: 'child:rg1:ref1' });
    expect(navigateTree(rows, 'child:rg1:ref2', 'ArrowDown')).toEqual({ focus: 'rg2' });
    expect(navigateTree(rows, 'rg2', 'ArrowUp')).toEqual({ focus: 'child:rg1:ref2' });
  });

  it('ArrowRight expands then enters the children; ArrowLeft collapses or climbs to the layer', () => {
    expect(navigateTree(flatten(new Set(['rg1'])), 'rg1', 'ArrowRight')).toEqual({
      expandChildren: 'rg1',
      expanded: true,
    });
    expect(navigateTree(flatten(), 'rg1', 'ArrowRight')).toEqual({ focus: 'child:rg1:ref1' });
    expect(navigateTree(flatten(), 'rg1', 'ArrowLeft')).toEqual({ expandChildren: 'rg1', expanded: false });
    expect(navigateTree(flatten(), 'child:rg1:ref2', 'ArrowLeft')).toEqual({ focus: 'rg1' });
    expect(navigateTree(flatten(), 'child:rg1:ref1', 'ArrowRight')).toBeNull();
    expect(navigateTree(flatten(), 'rg2', 'ArrowRight')).toBeNull();
    expect(navigateTree(flatten(new Set(['rg1'])), 'rg1', 'ArrowLeft')).toEqual({ focus: 'header:regional_guidance' });
  });
});

describe('group-owned adjustment rows', () => {
  const stack = (): CanvasAdjustmentEntry[] => [
    { brightness: 0.2, contrast: 0, id: 'ga1', isEnabled: true, type: 'brightness-contrast' },
    { id: 'ga2', isEnabled: true, type: 'invert' },
  ];
  const groupStacks = (expandedGroups: ReadonlySet<string>) =>
    buildLayerStackRows(
      documentFrom([
        groupContract('g1', [layerContract('m1'), layerContract('m2'), layerContract('m3')], {
          adjustments: stack(),
        }),
      ]).stacks,
      expandedGroups
    );
  const flattenGroup = (expandedGroups: ReadonlySet<string> = new Set(['g1'])) =>
    flattenPanelRows(groupStacks(expandedGroups), [], open, {
      collapsedLayerIds: new Set(),
      rowsFor: (row) => projectLayerChildRows(row.vm),
    });

  it("folds a group's rows with its subtree: a collapsed group renders neither", () => {
    expect(keys(flattenGroup(new Set()))).toEqual(['header:raster', 'g1']);
    expect(keys(flattenGroup())).toEqual(['header:raster', 'g1', 'child:g1:ga1', 'child:g1:ga2', 'm1', 'm2', 'm3']);
  });

  it('announces rows and member nodes as ONE combined ARIA set under the group', () => {
    const rows = flattenGroup();
    const children = rows.filter((row) => row.kind === 'child');
    expect(children.map((row) => (row.kind === 'child' ? row.ariaSetSize : 0))).toEqual([5, 5]);
    const members = rows.filter((row) => row.kind === 'node' && row.key.startsWith('m'));
    expect(members.map((row) => (row.kind === 'node' ? [row.ariaPosInSet, row.ariaSetSize] : null))).toEqual([
      [3, 5],
      [4, 5],
      [5, 5],
    ]);
  });

  it('ArrowRight expands the subtree; ArrowLeft collapses it — never a separate rows fold', () => {
    const collapsedRows = flattenGroup(new Set());
    expect(navigateTree(collapsedRows, 'g1', 'ArrowRight')).toEqual({ expand: 'g1', expanded: true });
    const expandedRows = flattenGroup();
    expect(navigateTree(expandedRows, 'g1', 'ArrowRight')).toEqual({ focus: 'child:g1:ga1' });
    expect(navigateTree(expandedRows, 'g1', 'ArrowLeft')).toEqual({ expand: 'g1', expanded: false });
  });
});

describe('tree guides', () => {
  const invert = (id: string): CanvasAdjustmentEntry => ({ id, isEnabled: true, type: 'invert' });
  /** top[a1] · g1[ga1]{ m1[x1, x2], m3, g2{ deep } } · bottom */
  const treeStacks = (expandedGroups: ReadonlySet<string>, filter?: string) =>
    buildLayerStackRows(
      documentFrom([
        layerContract('top', 'raster', { adjustments: [invert('a1')] }),
        groupContract(
          'g1',
          [
            layerContract('m1', 'raster', { adjustments: [invert('x1'), invert('x2')] }),
            layerContract('m3'),
            groupContract('g2', [layerContract('deep')]),
          ],
          { adjustments: [invert('ga1')] }
        ),
        layerContract('bottom'),
      ]).stacks,
      expandedGroups,
      filter
    );
  const guidesOf = (rows: readonly PanelRow[]) =>
    Object.fromEntries(rows.flatMap((row) => (row.kind === 'header' ? [] : [[row.key, row.guides]])));

  it('branches each visible child from its parent, ending the line at the last one', () => {
    const rows = flattenPanelRows(treeStacks(new Set(['g1', 'g2'])), [], open, {
      collapsedLayerIds: new Set(),
      rowsFor: (row) => projectLayerChildRows(row.vm),
    });
    expect(guidesOf(rows)).toEqual({
      bottom: '',
      'child:g1:ga1': '├',
      'child:m1:x1': '│├',
      'child:m1:x2': '│└',
      'child:top:a1': '└',
      deep: ' └',
      g1: '╷',
      g2: '└╷',
      m1: '├╷',
      m3: '├',
      top: '╷',
    });
    // Rows follow tree order, so every line runs unbroken from its stem to its last branch.
    expect(keys(rows)).toEqual([
      'header:raster',
      'top',
      'child:top:a1',
      'g1',
      'child:g1:ga1',
      'm1',
      'child:m1:x1',
      'child:m1:x2',
      'm3',
      'g2',
      'deep',
      'bottom',
    ]);
  });

  it('draws nothing for children a collapsed group or layer hides', () => {
    const rows = flattenPanelRows(treeStacks(new Set(['g1'])), [], open, {
      collapsedLayerIds: new Set(['m1', 'top']),
      rowsFor: (row) => projectLayerChildRows(row.vm),
    });
    expect(guidesOf(rows)).toEqual({
      bottom: '',
      'child:g1:ga1': '├',
      g1: '╷',
      g2: '└',
      m1: '├',
      m3: '├',
      top: '',
    });
  });

  it('follows the rows a filter keeps: matches under their opened ancestors', () => {
    expect(guidesOf(flattenPanelRows(treeStacks(new Set(), 'deep'), [], open))).toEqual({
      deep: ' └',
      g1: '╷',
      g2: '└╷',
    });
    expect(guidesOf(flattenPanelRows(treeStacks(new Set(), 'm'), [], open))).toEqual({
      bottom: '',
      g1: '╷',
      m1: '├',
      m3: '└',
    });
  });

  it('starts every stack afresh: top-level rows never branch from their stack header', () => {
    const rows = flattenPanelRows(stacks(), [], open, {
      collapsedLayerIds: new Set(),
      rowsFor: (row) => projectLayerChildRows(row.vm),
    });
    expect(guidesOf(rows)).toEqual({
      'child:rg1:ref1': '├',
      'child:rg1:ref2': '└',
      r1: '',
      rg1: '╷',
      rg2: '',
    });
  });
});
