import type { CanvasInpaintMaskLayerContract, RegionalGuidanceReferenceImage } from '@workbench/canvas-engine/api';

import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { readLayerPanelState, setLayerChildrenCollapsed } from '@workbench/layerPanelState';
import { commitPreparedEdit, type PreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  addMaskGuidance,
  clearLayerChildSelection,
  getLayerChildSelection,
  reconcileLayerChildSelection,
  selectLayerChild,
} from './layerChildSelection';

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

const documentWith = (refs: RegionalGuidanceReferenceImage[], selectedLayerId: string | null = 'rg1') =>
  documentFrom([layerContract('rg1', 'regional_guidance', { referenceImages: refs })], selectedLayerId);

describe('layer child selection', () => {
  beforeEach(() => clearLayerChildSelection());

  it('survives reconciliation while the owner is selected and the item exists', () => {
    selectLayerChild('p', 'rg1', 'ref1');
    reconcileLayerChildSelection('p', documentWith([referenceImage('ref1')]));
    expect(getLayerChildSelection()).toMatchObject({ itemId: 'ref1', layerId: 'rg1' });
  });

  it('clears when the owner loses document selection or the item disappears', () => {
    selectLayerChild('p', 'rg1', 'ref1');
    reconcileLayerChildSelection('p', documentWith([referenceImage('ref1')], 'other'));
    expect(getLayerChildSelection()).toBeNull();

    selectLayerChild('p', 'rg1', 'ref1');
    reconcileLayerChildSelection('p', documentWith([referenceImage('ref2')]));
    expect(getLayerChildSelection()).toBeNull();
  });

  it("leaves another project's selection alone", () => {
    selectLayerChild('other', 'rg1', 'ref1');
    reconcileLayerChildSelection('p', documentWith([], null));
    expect(getLayerChildSelection()).toMatchObject({ projectId: 'other' });
  });
});

describe('addMaskGuidance', () => {
  beforeEach(() => clearLayerChildSelection());

  it('adds empty guidance as one undo step, then selects the mask and its new row with the child rows open, once', async () => {
    const stub = createStructuralEngineStub({ layers: [layerContract('m1', 'inpaint_mask')] });
    const commit: PreparedCommit = (label, prepare) => commitPreparedEdit(stub.engine as never, label, prepare);
    const dispatch = vi.fn(() => true);
    const add = () =>
      addMaskGuidance({ commit, dispatch, label: 'Add regional guidance', layerId: 'm1', projectId: 'p' });
    const guidance = () => (getDocumentLayer(stub.document(), 'm1') as CanvasInpaintMaskLayerContract).guidance;
    setLayerChildrenCollapsed('p', 'm1', 'm1', true);

    expect(add()).toBe(true);

    expect(guidance()).toEqual({ autoNegative: false, isEnabled: true, negativePrompt: null, positivePrompt: null });
    expect(stub.history.entries().past).toEqual(['Add regional guidance']);
    expect(dispatch.mock.calls).toEqual([[{ id: 'm1', type: 'setCanvasSelectedLayer' }]]);
    expect(getLayerChildSelection()).toEqual({ itemId: 'guidance', layerId: 'm1', projectId: 'p' });
    expect(readLayerPanelState('p', 'm1')).toMatchObject({
      collapsedChildLayerIds: [],
      focusId: 'child:m1:guidance',
      selectedIds: ['m1'],
    });

    clearLayerChildSelection();
    expect(add()).toBe(false);
    expect(stub.commits).toHaveLength(1);
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(getLayerChildSelection()).toBeNull();

    await stub.engine.history.undo();
    expect(guidance()).toBeUndefined();
  });
});
