import { describe, expect, it } from 'vitest';

import { createLayerPanelSelection, reconcileLayerPanelSelection, selectLayerInPanel } from './layerGroups';

const ids = ['a', 'b', 'c', 'd'];
const plain = { additive: false, range: false };
const toggle = { additive: true, range: false };
const range = { additive: false, range: true };

describe('layer panel selection', () => {
  it('replaces selection on a plain click', () => {
    const initial = createLayerPanelSelection('project', 'a');
    expect(selectLayerInPanel(initial, 'c', ids, plain)).toMatchObject({
      anchorId: 'c',
      primaryId: 'c',
      selectedIds: ['c'],
    });
  });

  it('toggles layers with Ctrl/Cmd while keeping a valid primary', () => {
    const initial = createLayerPanelSelection('project', 'b');
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
    const initial = createLayerPanelSelection('project', 'b');
    expect(selectLayerInPanel(initial, 'd', ids, range)).toMatchObject({
      anchorId: 'b',
      primaryId: 'd',
      selectedIds: ['b', 'c', 'd'],
    });
  });

  it('adds a Shift range when Ctrl/Cmd is held too', () => {
    const initial = selectLayerInPanel(createLayerPanelSelection('project', 'a'), 'd', ids, toggle);
    const next = selectLayerInPanel(initial, 'b', ids, { additive: true, range: true });
    expect(next.selectedIds).toEqual(['a', 'b', 'c', 'd']);
  });

  it('uses only rendered rows when a Shift range crosses collapsed groups', () => {
    const initial = createLayerPanelSelection('project', 'a');
    expect(selectLayerInPanel(initial, 'd', ['a', 'd'], range).selectedIds).toEqual(['a', 'd']);
  });

  it('prunes removed secondaries without collapsing an unchanged primary', () => {
    const multi = selectLayerInPanel(createLayerPanelSelection('project', 'a'), 'c', ids, toggle);
    expect(reconcileLayerPanelSelection(multi, 'project', ['a', 'b', 'd'], 'a')).toMatchObject({
      primaryId: 'a',
      selectedIds: ['a'],
    });
  });

  it('collapses to a new primary selected outside the panel and resets between projects', () => {
    const multi = selectLayerInPanel(createLayerPanelSelection('project', 'a'), 'c', ids, toggle);
    expect(reconcileLayerPanelSelection(multi, 'project', ids, 'd').selectedIds).toEqual(['d']);
    expect(reconcileLayerPanelSelection(multi, 'other-project', ids, 'a').selectedIds).toEqual(['a']);
  });

  it('does not resurrect secondaries after an A to B to A project round trip', () => {
    let selection = selectLayerInPanel(createLayerPanelSelection('project-a', 'a'), 'c', ids, toggle);
    selection = reconcileLayerPanelSelection(selection, 'project-b', ids, 'b');
    selection = reconcileLayerPanelSelection(selection, 'project-a', ids, 'a');
    expect(selection.selectedIds).toEqual(['a']);
  });
});
