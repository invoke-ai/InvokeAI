import type { CommitGeneratedImageOptions } from '@workbench/canvas-engine/capabilities';
import type { CanvasControlLayerContract, CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it } from 'vitest';

import { GeneratedResultController } from './generatedResultController';
import { createLayerOperationHarness } from './layerOperationHarness.testing';

const DEPTH = { settings: {}, type: 'depth_anything_depth_estimation' };

const derivedControl: CanvasControlLayerContract = {
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  controlType: 'depth',
  detection: DEPTH,
  filter: DEPTH,
  id: 'control',
  isEnabled: true,
  isLocked: false,
  name: 'Depth 1',
  opacity: 1,
  original: { image: { height: 16, imageName: 'photo', width: 16 }, type: 'image' },
  source: { bitmap: { height: 16, imageName: 'map', width: 16 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 3, y: 4 },
  type: 'control',
  withTransparencyEffect: true,
};

const createHarness = () => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 16, width: 16, x: 0, y: 0 },
    height: 16,
    selectedLayerId: 'control',
    stacks: stacksFrom([derivedControl]),
    version: 3,
    width: 16,
  };
  const h = createLayerOperationHarness(document);
  h.layers.getOrCreateRect('control', { height: 16, width: 16, x: 0, y: 0 });
  const controller = new GeneratedResultController({
    captureCache: (target) => {
      const entry = h.layers.get(target.id);
      return entry ? { pixels: h.backend.createSurface(16, 16), rect: { ...entry.rect } } : null;
    },
    clearPreview: () => undefined,
    ctx: h.ctx,
    decodeImage: () =>
      Promise.resolve({ decodedHeight: 16, decodedWidth: 16, status: 'ok', surface: h.backend.createSurface(16, 16) }),
    discardPersisted: () => undefined,
    getMainModelBase: () => 'sdxl',
    needsPixelPersistence: () => true,
  });
  const commit = (target: CommitGeneratedImageOptions['target']) =>
    controller.commit({
      guard: { layerId: 'control' } as CommitGeneratedImageOptions['guard'],
      image: { height: 16, imageName: 'workflow-result', width: 16 },
      origin: { x: 0, y: 0 },
      target,
    });
  return { ...h, commit };
};

describe('GeneratedResultController control derivation', () => {
  it('replacing a derived control map keeps the result as the image itself', async () => {
    const h = createHarness();

    await expect(h.commit('replace')).resolves.toEqual({ layerId: 'control', status: 'committed' });

    const replaced = getDocumentLayer(h.document(), 'control');
    expect(replaced).toMatchObject({ controlType: 'depth', detection: null });
    expect(replaced).not.toHaveProperty('original');
    // Not map-only: the result itself is what a later detection runs on.
    expect(replaced).not.toHaveProperty('filter');
    await h.history.undo();
    expect(getDocumentLayer(h.document(), 'control')).toEqual(derivedControl);
  });

  it('copies a result to an untyped control layer', async () => {
    const h = createHarness();

    const result = await h.commit('copy-control');

    const copy = getDocumentLayer(h.document(), 'layerId' in result ? result.layerId : null);
    expect(copy?.type).toBe('control');
    for (const key of ['controlType', 'detection', 'original', 'originalTransform']) {
      expect(copy).not.toHaveProperty(key);
    }
  });
});
