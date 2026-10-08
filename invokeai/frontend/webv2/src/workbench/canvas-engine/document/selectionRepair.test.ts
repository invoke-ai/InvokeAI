import type { CanvasStackForests } from '@workbench/canvas-engine/contracts';

import {
  documentFrom,
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { applyCanvasProjectMutation, type CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it } from 'vitest';

import { removeNodes } from './documentTree';
import { repairSelectedLayerId, selectionAfterAcceptedResult } from './selectionRepair';

const layer = layerContract;
const group = groupContract;

const previous = (): CanvasStackForests => stacksFrom(['a', 'b', 'c', 'd'].map((id) => layer(id)));
const without = (...ids: string[]): CanvasStackForests => removeNodes(previous(), new Set(ids));

describe('repairSelectedLayerId', () => {
  it.each([
    ['keeps a surviving selection', without('a'), 'b', previous(), 'b'],
    ['keeps null', previous(), null, previous(), null],
    ['moves below first', without('b'), 'b', previous(), 'c'],
    ['moves above when nothing survives below', without('c', 'd'), 'd', previous(), 'b'],
    ['moves through a removed run', without('b', 'c'), 'b', previous(), 'd'],
    ['falls back to the top without a previous order', without('b'), 'b', undefined, 'a'],
    ['falls back to the top when the selection was never present', without('b'), 'x', previous(), 'a'],
    ['clears on an empty document', stacksFrom([]), 'b', previous(), null],
    ['clears on an empty document without a previous order', stacksFrom([]), 'b', undefined, null],
  ])('%s', (_label, stacks, selected, previousStacks, expected) => {
    expect(repairSelectedLayerId(stacks, selected, previousStacks)).toBe(expected);
  });

  it('prefers a sibling, then the parent, then the nearest node of the same stack', () => {
    /** raster: r1, g1[r2, r3], r4 · control: c1 */
    const tree = stacksFrom([
      layer('r1'),
      group('g1', [layer('r2'), layer('r3')]),
      layer('r4'),
      layer('c1', 'control'),
    ]);

    expect(repairSelectedLayerId(removeNodes(tree, new Set(['r2'])), 'r2', tree)).toBe('r3');
    expect(repairSelectedLayerId(removeNodes(tree, new Set(['r2', 'r3'])), 'r2', tree)).toBe('g1');
    expect(repairSelectedLayerId(removeNodes(tree, new Set(['g1'])), 'r2', tree)).toBe('r4');
    expect(repairSelectedLayerId(removeNodes(tree, new Set(['g1', 'r4'])), 'g1', tree)).toBe('r1');
    expect(repairSelectedLayerId(removeNodes(tree, new Set(['r1', 'g1', 'r4'])), 'g1', tree)).toBe('c1');
  });

  it('selects the top node of the top-first stacks when there is no previous order', () => {
    const tree = stacksFrom([layer('r1'), layer('i1', 'inpaint_mask')]);
    expect(repairSelectedLayerId(tree, 'ghost')).toBe('i1');
  });
});

describe('selection repair through the reducer', () => {
  const nodes = () => [
    layer('i1', 'inpaint_mask'),
    layer('r1'),
    layer('c1', 'control'),
    group('g1', [layer('r2'), layer('r3')]),
    layer('r4'),
  ];
  const reduce = (selectedLayerId: string | null, mutation: CanvasProjectMutation) => {
    const project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
      document: documentFrom(nodes(), selectedLayerId),
      type: 'replaceCanvasDocument',
    });
    return applyCanvasProjectMutation(project, mutation).canvas.document.selectedLayerId;
  };
  const paint = { bitmap: null, type: 'paint' as const };

  it.each<[string, string | null, CanvasProjectMutation, string | null]>([
    ['remove keeps a surviving selection', 'r2', { ids: ['r1'], type: 'removeCanvasLayers' }, 'r2'],
    ['remove moves to the sibling below', 'r2', { ids: ['r2'], type: 'removeCanvasLayers' }, 'r3'],
    [
      'remove moves to the parent when the group empties',
      'r3',
      { ids: ['r2', 'r3'], type: 'removeCanvasLayers' },
      'g1',
    ],
    ['remove of a selected group moves to a stack neighbour', 'g1', { ids: ['g1'], type: 'removeCanvasLayers' }, 'r4'],
    ['remove falls back across stacks when the stack empties', 'c1', { ids: ['c1'], type: 'removeCanvasLayers' }, 'r1'],
    [
      'merge down moves the selection onto the merged layer',
      'r2',
      { source: paint, type: 'mergeCanvasLayersDown', upperLayerId: 'r2' },
      'r3',
    ],
    [
      'merge down keeps a selection elsewhere',
      'i1',
      { source: paint, type: 'mergeCanvasLayersDown', upperLayerId: 'r2' },
      'i1',
    ],
    [
      'replacement keeps a selection the new document holds',
      'r1',
      { document: documentFrom([layer('r1')], 'r1'), type: 'replaceCanvasDocument' },
      'r1',
    ],
    [
      'replacement selects the top when its selection is absent',
      'r1',
      { document: documentFrom([layer('x'), layer('y')], 'nope'), type: 'replaceCanvasDocument' },
      'x',
    ],
    [
      'an emptied document clears the selection',
      'r1',
      { ids: ['i1', 'r1', 'c1', 'g1', 'r4'], type: 'removeCanvasLayers' },
      null,
    ],
    [
      'an empty replacement clears the selection',
      'r1',
      { document: documentFrom([], 'r1'), type: 'replaceCanvasDocument' },
      null,
    ],
  ])('%s', (_label, selected, mutation, expected) => {
    expect(reduce(selected, mutation)).toBe(expected);
  });
});

describe('selectionAfterAcceptedResult', () => {
  /** raster: r1, g-r[r2] · control: c1, c-off · regional: g-o[rg1], g-off[rg2], g-lock[rg3] · inpaint: m1, m-lock */
  const stacks = stacksFrom([
    layer('r1'),
    group('g-r', [layer('r2')]),
    layer('c1', 'control'),
    layer('c-off', 'control', { isEnabled: false }),
    group('g-o', [layer('rg1', 'regional_guidance')]),
    group('g-off', [layer('rg2', 'regional_guidance')], { isEnabled: false }),
    group('g-lock', [layer('rg3', 'regional_guidance')], { isLocked: true }),
    layer('m1', 'inpaint_mask'),
    layer('m-lock', 'inpaint_mask', { isLocked: true }),
  ]);

  it.each([
    ['keeps a selected inpaint mask', 'm1', 'm1'],
    ['keeps a selected regional mask inside an editable group', 'rg1', 'rg1'],
    ['keeps a selected control layer', 'c1', 'c1'],
    ['selects the result over a locked mask', 'm-lock', 'accepted'],
    ['selects the result over a disabled control layer', 'c-off', 'accepted'],
    ['selects the result over a mask in a disabled group', 'rg2', 'accepted'],
    ['selects the result over a mask in a locked group', 'rg3', 'accepted'],
    ['selects the result over an overlay group, which no stroke can paint', 'g-o', 'accepted'],
    ['selects the result over a raster leaf it now covers', 'r1', 'accepted'],
    ['selects the result over a raster group', 'g-r', 'accepted'],
    ['selects the result when nothing was selected', null, 'accepted'],
    ['selects the result when the selection no longer exists', 'gone', 'accepted'],
  ])('%s', (_label, selected, expected) => {
    expect(selectionAfterAcceptedResult(stacks, selected, 'accepted')).toBe(expected);
  });
});
