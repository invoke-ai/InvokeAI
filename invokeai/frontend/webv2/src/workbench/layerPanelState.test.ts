import type { CanvasLayerContract } from '@workbench/canvas-engine/api';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { indexStacks } from '@workbench/canvas-engine/document/documentIndex';
import { beforeEach, describe, expect, it } from 'vitest';

import {
  clearLayerPanelStates,
  createLayerPanelState,
  publishLayerPanelSelection,
  readLayerPanelState,
  reconcileLayerPanelState,
  isSameLayerPanelState,
  reconcileLayerPanelStates,
  selectLayerInPanel,
  setLayerPanelFilter,
  setLayerPanelFocus,
  toggleLayerStackCollapsed,
  type LayerPanelProjectView,
} from './layerPanelState';

const ids = ['a', 'b', 'c', 'd'];
const indexOf = (nodeIds: readonly string[]) => indexStacks(stacksFrom(nodeIds.map((id) => layerContract(id))));
const plain = { additive: false, range: false };
const toggle = { additive: true, range: false };
const range = { additive: false, range: true };

describe('layer panel selection', () => {
  it('replaces selection on a plain click', () => {
    const initial = createLayerPanelState('project', 'a');
    expect(selectLayerInPanel(initial, 'c', ids, plain)).toMatchObject({
      anchorId: 'c',
      primaryId: 'c',
      selectedIds: ['c'],
    });
  });

  it('toggles layers with Ctrl/Cmd while keeping a valid primary', () => {
    const initial = createLayerPanelState('project', 'b');
    const added = selectLayerInPanel(initial, 'd', ids, toggle);
    expect(added).toMatchObject({ primaryId: 'd', selectedIds: ['b', 'd'] });
    expect(selectLayerInPanel(added, 'b', ids, toggle)).toMatchObject({
      primaryId: 'd',
      selectedIds: ['d'],
    });
    expect(selectLayerInPanel(added, 'd', ids, toggle)).toMatchObject({
      primaryId: 'b',
      selectedIds: ['b'],
    });
  });

  it('selects a contiguous Shift range from the stable anchor', () => {
    const initial = createLayerPanelState('project', 'b');
    expect(selectLayerInPanel(initial, 'd', ids, range)).toMatchObject({
      anchorId: 'b',
      primaryId: 'd',
      selectedIds: ['b', 'c', 'd'],
    });
  });

  it('adds a Shift range when Ctrl/Cmd is held too', () => {
    const initial = selectLayerInPanel(createLayerPanelState('project', 'a'), 'd', ids, toggle);
    const next = selectLayerInPanel(initial, 'b', ids, { additive: true, range: true });
    expect(next.selectedIds).toEqual(['a', 'b', 'c', 'd']);
  });

  it('uses only rendered rows when a Shift range crosses collapsed groups', () => {
    const initial = createLayerPanelState('project', 'a');
    expect(selectLayerInPanel(initial, 'd', ['a', 'd'], range).selectedIds).toEqual(['a', 'd']);
  });

  it('keeps selected rows hidden inside a collapsed group through an additive range', () => {
    const initial = { ...createLayerPanelState('project', 'a'), anchorId: 'c', selectedIds: ['a', 'c'] };
    expect(selectLayerInPanel(initial, 'd', ['c', 'd'], { additive: true, range: true }).selectedIds).toEqual([
      'c',
      'd',
      'a',
    ]);
    expect(selectLayerInPanel(initial, 'd', ['c', 'd'], range).selectedIds).toEqual(['c', 'd']);
  });

  it('falls back to the top-most selected row when the primary is toggled off', () => {
    let state = createLayerPanelState('project', 'c');
    state = selectLayerInPanel(state, 'a', ids, toggle);
    state = selectLayerInPanel(state, 'b', ids, toggle);
    expect(state).toMatchObject({ primaryId: 'b', selectedIds: ['a', 'b', 'c'] });
    expect(selectLayerInPanel(state, 'b', ids, toggle)).toMatchObject({ primaryId: 'a', selectedIds: ['a', 'c'] });
  });

  it('drops duplicate ids while reconciling', () => {
    const state = { ...createLayerPanelState('project', 'a'), selectedIds: ['a', 'a', 'b'] };
    expect(reconcileLayerPanelState(state, 'project', indexOf(ids), 'a').selectedIds).toEqual(['a', 'b']);
  });

  it('prunes removed secondaries without collapsing an unchanged primary', () => {
    const multi = { ...createLayerPanelState('project', 'a'), anchorId: 'c', selectedIds: ['a', 'b', 'c'] };
    expect(reconcileLayerPanelState(multi, 'project', indexOf(['a', 'b', 'd']), 'a')).toEqual({
      ...multi,
      anchorId: 'a',
      selectedIds: ['a', 'b'],
    });
    expect(reconcileLayerPanelState(multi, 'project', indexOf(ids), 'a')).toBe(multi);
  });

  it('carries collapsed stacks through a same-project primary change only', () => {
    const collapsed = { ...createLayerPanelState('project', 'a'), collapsedStacks: ['control' as const] };
    expect(reconcileLayerPanelState(collapsed, 'project', indexOf(ids), 'b')).toMatchObject({
      collapsedStacks: ['control'],
      primaryId: 'b',
    });
    expect(reconcileLayerPanelState(collapsed, 'other', indexOf(ids), 'b').collapsedStacks).toEqual([]);
  });

  it('opens the collapsed stack of a primary selected outside the panel', () => {
    const index = indexStacks(stacksFrom([layerContract('c', 'control'), layerContract('a')]));
    const collapsed = {
      ...createLayerPanelState('p', 'a'),
      collapsedStacks: ['control' as const, 'inpaint_mask' as const],
    };
    expect(reconcileLayerPanelState(collapsed, 'p', index, 'c').collapsedStacks).toEqual(['inpaint_mask']);
    const stale = { ...collapsed, expandedGroupIds: ['gone'] };
    expect(reconcileLayerPanelState(stale, 'p', index, 'c').expandedGroupIds).toEqual([]);
    expect(reconcileLayerPanelState(collapsed, 'p', index, 'a')).toBe(collapsed);
  });

  it('keeps a child-row focus while its owner exists and drops it with the owner', () => {
    const focused = { ...createLayerPanelState('project', 'a'), focusId: 'child:b:ref1' };
    expect(reconcileLayerPanelState(focused, 'project', indexOf(ids), 'a')).toBe(focused);
    expect(reconcileLayerPanelState(focused, 'project', indexOf(['a', 'c']), 'a').focusId).toBe('a');
  });

  it('carries child-row collapse through a primary change and prunes removed owners', () => {
    const collapsed = { ...createLayerPanelState('project', 'a'), collapsedChildLayerIds: ['a', 'gone'] };
    expect(reconcileLayerPanelState(collapsed, 'project', indexOf(ids), 'b').collapsedChildLayerIds).toEqual(['a']);
    expect(reconcileLayerPanelState(collapsed, 'project', indexOf(ids), 'a').collapsedChildLayerIds).toEqual(['a']);
    expect(reconcileLayerPanelState(collapsed, 'other', indexOf(ids), 'b').collapsedChildLayerIds).toEqual([]);
  });

  it('collapses to a new primary selected outside the panel and resets between projects', () => {
    const multi = selectLayerInPanel(createLayerPanelState('project', 'a'), 'c', ids, toggle);
    expect(reconcileLayerPanelState(multi, 'project', indexOf(ids), 'd').selectedIds).toEqual(['d']);
    expect(reconcileLayerPanelState(multi, 'other-project', indexOf(ids), 'a').selectedIds).toEqual(['a']);
  });

  it('does not resurrect secondaries after an A to B to A project round trip', () => {
    let selection = selectLayerInPanel(createLayerPanelState('project-a', 'a'), 'c', ids, toggle);
    selection = reconcileLayerPanelState(selection, 'project-b', indexOf(ids), 'b');
    selection = reconcileLayerPanelState(selection, 'project-a', indexOf(ids), 'a');
    expect(selection.selectedIds).toEqual(['a']);
  });
});

describe('layer panel state store', () => {
  beforeEach(() => clearLayerPanelStates());

  const project = (id: string, layerIds: readonly string[], selectedLayerId: string | null): LayerPanelProjectView => ({
    canvas: {
      document: {
        stacks: stacksFrom(layerIds.map((layerId) => ({ id: layerId }) as CanvasLayerContract)),
        selectedLayerId,
      },
    },
    id,
  });

  it('keeps selection and collapsed stacks per project', () => {
    publishLayerPanelSelection({ primaryId: 'a', projectId: 'p1', selectedIds: ['a', 'c'] });
    toggleLayerStackCollapsed('p1', 'a', 'control');
    publishLayerPanelSelection({ primaryId: 'x', projectId: 'p2', selectedIds: ['x', 'y'] });

    expect(readLayerPanelState('p1', 'a')).toMatchObject({ collapsedStacks: ['control'], selectedIds: ['a', 'c'] });
    expect(readLayerPanelState('p2', 'x')).toMatchObject({ collapsedStacks: [], selectedIds: ['x', 'y'] });
  });

  it('keeps collapsed stacks but drops secondaries when the primary changed outside the panel', () => {
    publishLayerPanelSelection({ primaryId: 'a', projectId: 'p1', selectedIds: ['a', 'c'] });
    toggleLayerStackCollapsed('p1', 'a', 'raster');

    expect(readLayerPanelState('p1', 'b')).toMatchObject({
      collapsedStacks: ['raster'],
      primaryId: 'b',
      selectedIds: ['b'],
    });
    expect(readLayerPanelState('p3', null)).toMatchObject({ collapsedStacks: [], selectedIds: [] });
  });

  it('toggles a collapsed stack on and off', () => {
    toggleLayerStackCollapsed('p1', null, 'inpaint_mask');
    expect(readLayerPanelState('p1', null).collapsedStacks).toEqual(['inpaint_mask']);
    toggleLayerStackCollapsed('p1', null, 'inpaint_mask');
    expect(readLayerPanelState('p1', null).collapsedStacks).toEqual([]);
  });

  it('reconciles every project against its document and forgets closed projects', () => {
    publishLayerPanelSelection({ primaryId: 'a', projectId: 'p1', selectedIds: ['a', 'c'] });
    publishLayerPanelSelection({ primaryId: 'x', projectId: 'p2', selectedIds: ['x', 'y'] });

    reconcileLayerPanelStates([project('p1', ['a', 'b'], 'a')]);

    expect(readLayerPanelState('p1', 'a').selectedIds).toEqual(['a']);
    expect(readLayerPanelState('p2', 'x').selectedIds).toEqual(['x']);
  });

  it('returns the same state for a selection that already matches its document', () => {
    publishLayerPanelSelection({ anchorId: 'c', primaryId: 'a', projectId: 'p1', selectedIds: ['a', 'c'] });
    toggleLayerStackCollapsed('p1', 'a', 'control');
    publishLayerPanelSelection({ primaryId: 'x', projectId: 'p2', selectedIds: ['x', 'y'] });
    const p1 = project('p1', ['a', 'b', 'c'], 'a');
    const before = readLayerPanelState('p1', 'a');

    reconcileLayerPanelStates([p1, project('p2', ['x', 'y'], 'x')]);
    expect(readLayerPanelState('p1', 'a')).toBe(before);
    expect(before).toMatchObject({ anchorId: 'c', collapsedStacks: ['control'], selectedIds: ['a', 'c'] });

    reconcileLayerPanelStates([p1, project('p2', ['x'], 'x')]);
    expect(readLayerPanelState('p1', 'a')).toBe(before);
    expect(readLayerPanelState('p2', 'x').selectedIds).toEqual(['x']);
  });

  it('compares states by content so a rebuilt read stays stable for subscribers', () => {
    toggleLayerStackCollapsed('p1', 'a', 'control');
    expect(isSameLayerPanelState(readLayerPanelState('p1', 'b'), readLayerPanelState('p1', 'b'))).toBe(true);
    expect(isSameLayerPanelState(readLayerPanelState('p1', 'a'), readLayerPanelState('p1', 'b'))).toBe(false);
  });
});

describe('revealing an external primary', () => {
  it('expands every group above a primary that arrived from outside the panel', () => {
    const index = indexStacks(
      stacksFrom([groupContract('g', [groupContract('h', [layerContract('a')])]), layerContract('b')])
    );
    expect(reconcileLayerPanelState(createLayerPanelState('project', 'b'), 'project', index, 'a')).toMatchObject({
      expandedGroupIds: ['g', 'h'],
      primaryId: 'a',
      selectedIds: ['a'],
    });
    const collapsedByUser = createLayerPanelState('project', 'a');
    expect(reconcileLayerPanelState(collapsedByUser, 'project', index, 'a')).toBe(collapsedByUser);
  });
});

describe('focus and filter', () => {
  beforeEach(() => clearLayerPanelStates());

  it('keeps a roving focus that follows the primary until a row is focused explicitly', () => {
    const index = indexStacks(stacksFrom([layerContract('a'), layerContract('b')]));
    publishLayerPanelSelection({ primaryId: 'a', projectId: 'p1', selectedIds: ['a'] });
    expect(readLayerPanelState('p1', 'a').focusId).toBe('a');
    setLayerPanelFocus('p1', 'a', 'b');
    expect(readLayerPanelState('p1', 'a').focusId).toBe('b');
    const gone = reconcileLayerPanelState(
      readLayerPanelState('p1', 'a'),
      'p1',
      indexStacks(stacksFrom([layerContract('a')])),
      'a'
    );
    expect(gone.focusId).toBe('a');
    expect(reconcileLayerPanelState(readLayerPanelState('p1', 'a'), 'p1', index, 'a').focusId).toBe('b');
  });

  it('carries the filter across primary changes and clears it on a project switch', () => {
    setLayerPanelFilter('p1', 'a', 'sky');
    expect(readLayerPanelState('p1', 'b')).toMatchObject({ filter: 'sky', primaryId: 'b' });
    const index = indexStacks(stacksFrom([layerContract('x')]));
    expect(reconcileLayerPanelState(readLayerPanelState('p1', 'b'), 'p2', index, 'x')).toMatchObject({
      filter: '',
    });
  });
});
