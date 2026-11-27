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
  reconcileLayerPanelStates,
  selectLayerInPanel,
  setLayerPanelFilter,
  toggleLayerStackCollapsed,
  type LayerPanelProjectView,
} from './layerPanelState';

const ids = ['a', 'b', 'c', 'd'];
const indexOf = (nodeIds: readonly string[]) => indexStacks(stacksFrom(nodeIds.map((id) => layerContract(id))));
const toggle = { additive: true, range: false };
const range = { additive: false, range: true };

describe('layer panel selection', () => {
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

  it('uses only rendered rows when a Shift range crosses collapsed groups', () => {
    const initial = createLayerPanelState('project', 'a');
    expect(selectLayerInPanel(initial, 'd', ['a', 'd'], range).selectedIds).toEqual(['a', 'd']);
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

  it('reconciles every project against its document and forgets closed projects', () => {
    publishLayerPanelSelection({ primaryId: 'a', projectId: 'p1', selectedIds: ['a', 'c'] });
    publishLayerPanelSelection({ primaryId: 'x', projectId: 'p2', selectedIds: ['x', 'y'] });

    reconcileLayerPanelStates([project('p1', ['a', 'b'], 'a')]);

    expect(readLayerPanelState('p1', 'a').selectedIds).toEqual(['a']);
    expect(readLayerPanelState('p2', 'x').selectedIds).toEqual(['x']);
  });
});

describe('revealing an external primary', () => {
  it('expands every group above a primary that arrived from outside the panel', () => {
    const index = indexStacks(
      stacksFrom([groupContract('g', [groupContract('h', [layerContract('a')])]), layerContract('b')])
    );
    expect(
      reconcileLayerPanelState(createLayerPanelState('project', 'b', { filter: 'b' }), 'project', index, 'a')
    ).toMatchObject({
      expandedGroupIds: ['g', 'h'],
      filter: '',
      primaryId: 'a',
      selectedIds: ['a'],
    });
    const collapsedByUser = createLayerPanelState('project', 'a');
    expect(reconcileLayerPanelState(collapsedByUser, 'project', index, 'a')).toBe(collapsedByUser);
  });
});

describe('focus and filter', () => {
  beforeEach(() => clearLayerPanelStates());

  it('carries the filter across primary changes and never into another project', () => {
    setLayerPanelFilter('p1', 'a', 'sky');
    expect(readLayerPanelState('p1', 'b')).toMatchObject({ filter: 'sky', primaryId: 'b' });
    const index = indexStacks(stacksFrom([layerContract('x')]));
    expect(reconcileLayerPanelState(readLayerPanelState('p1', 'b'), 'p2', index, 'x')).toMatchObject({
      filter: '',
    });
  });
});
