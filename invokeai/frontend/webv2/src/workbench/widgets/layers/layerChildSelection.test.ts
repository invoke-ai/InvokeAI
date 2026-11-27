import type { RegionalGuidanceReferenceImage } from '@workbench/canvas-engine/api';

import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { beforeEach, describe, expect, it } from 'vitest';

import {
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
