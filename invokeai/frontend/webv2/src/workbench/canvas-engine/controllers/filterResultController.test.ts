import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
} from '@workbench/canvas-engine/contracts';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it } from 'vitest';

import type { CommitRasterFilterOptions } from './filterResultController';

import { FilterResultController } from './filterResultController';
import { createLayerOperationHarness } from './layerOperationHarness.testing';

const DEPTH = { settings: {}, type: 'depth_anything_depth_estimation' };
const CANNY = { settings: { low_threshold: 100 }, type: 'canny_edge_detection' };
const RESULT = { height: 16, imageName: 'filtered', width: 16 };

const derivedControl: CanvasControlLayerContract = {
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: 'depth-model', weight: 0.8 },
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
  originalTransform: { rotation: 0, scaleX: 1, scaleY: 1, x: 2, y: 2 },
  source: { bitmap: { height: 16, imageName: 'map', width: 16 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
};

const rasterLayer: CanvasLayerContract = {
  blendMode: 'normal',
  id: 'raster',
  isEnabled: true,
  isLocked: false,
  name: 'Photo',
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};

const createHarness = () => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 16, width: 16, x: 0, y: 0 },
    height: 16,
    selectedLayerId: 'control',
    stacks: stacksFrom([rasterLayer, derivedControl]),
    version: 3,
    width: 16,
  };
  const h = createLayerOperationHarness(document);
  h.layers.getOrCreateRect('control', { height: 16, width: 16, x: 0, y: 0 });
  const controller = new FilterResultController({
    captureCache: (target) => {
      const entry = h.layers.get(target.id);
      return entry ? { pixels: h.backend.createSurface(16, 16), rect: { ...entry.rect } } : null;
    },
    ctx: h.ctx,
    decodeImage: () =>
      Promise.resolve({ decodedHeight: 16, decodedWidth: 16, status: 'ok', surface: h.backend.createSurface(16, 16) }),
    discardPersisted: () => undefined,
    getMainModelBase: () => 'sdxl',
    needsPixelPersistence: () => true,
  });
  const commit = (layerId: string, options: Partial<CommitRasterFilterOptions>) =>
    controller.commit({
      filter: CANNY,
      guard: { layerId } as CommitRasterFilterOptions['guard'],
      image: RESULT,
      mode: 'copy',
      rect: { height: 16, width: 16, x: 0, y: 0 },
      ...options,
    });
  return { ...h, commit };
};

const committedLayer = (h: ReturnType<typeof createHarness>, result: Awaited<ReturnType<typeof h.commit>>) => {
  expect(result.status).toBe('committed');
  return getDocumentLayer(h.document(), 'layerId' in result ? result.layerId : null);
};

describe('FilterResultController', () => {
  it('refuses a control source, whose map comes from detection, and leaves the document alone', async () => {
    const h = createHarness();
    const before = h.document();

    expect(await h.commit('control', { mode: 'replace' })).toEqual({ status: 'unsupported' });
    expect(await h.commit('control', { controlCopy: { controlType: 'edges' }, target: 'control' })).toEqual({
      status: 'unsupported',
    });
    expect(await h.commit('control', { target: 'raster' })).toEqual({ status: 'unsupported' });
    expect(h.document()).toBe(before);
  });

  it('copies a raster result to a new control layer, typed by the filter when the caller says so', async () => {
    const h = createHarness();

    const typed = committedLayer(
      h,
      await h.commit('raster', { controlCopy: { controlType: 'edges' }, target: 'control' })
    );
    expect(typed).toMatchObject({
      controlType: 'edges',
      detection: null,
      filter: CANNY,
      name: 'Photo filtered',
      source: { bitmap: RESULT, type: 'paint' },
      type: 'control',
    });

    const untyped = committedLayer(
      h,
      await h.commit('raster', { controlCopy: { controlType: null }, target: 'control' })
    );
    expect(untyped).toMatchObject({ adapter: { model: null }, filter: CANNY, type: 'control' });
    for (const key of ['controlType', 'detection', 'original', 'originalTransform']) {
      expect(untyped).not.toHaveProperty(key);
    }
  });
});
