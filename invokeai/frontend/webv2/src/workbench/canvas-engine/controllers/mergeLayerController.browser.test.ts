/** Real-canvas checks that Merge Down bakes each layer's own adjustment stack as the canvas displayed it. */

import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { MergeLayerController } from './mergeLayerController';

const NOISE: CanvasAdjustmentsContract = [
  {
    amount: 0.25,
    distribution: 'gaussian',
    grain: 'monochrome',
    id: 'grain',
    isEnabled: true,
    response: 'flat',
    seed: 21,
    size: 1,
    type: 'noise',
  },
];
const IDENTITY = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
const UPPER_RECT: Rect = { height: 18, width: 24, x: -3, y: 2 };
const LOWER_RECT: Rect = { height: 20, width: 20, x: 6, y: -2 };
/** Document region covering both layers before and after the merge. */
const VIEW: Rect = { height: 48, width: 48, x: -8, y: -8 };

const layer = (
  id: string,
  rect: Rect,
  adjustments: CanvasAdjustmentsContract | undefined,
  transform = IDENTITY
): CanvasRasterLayerContractV2 => ({
  ...(adjustments ? { adjustments } : {}),
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: { height: rect.height, imageName: `${id}.png`, width: rect.width }, offset: rect, type: 'paint' },
  transform,
  type: 'raster',
});

interface Scene {
  upperAdjustments?: CanvasAdjustmentsContract;
  lowerAdjustments?: CanvasAdjustmentsContract;
  upperTransform?: CanvasRasterLayerContractV2['transform'];
}

const mergeScene = (scene: Scene, displayed: boolean) => {
  const backend = createDomRasterBackend();
  const upper = layer('upper', UPPER_RECT, scene.upperAdjustments, scene.upperTransform);
  const lower = layer('lower', LOWER_RECT, scene.lowerAdjustments);
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 64, width: 64, x: 0, y: 0 },
    height: 64,
    selectedLayerId: 'upper',
    stacks: stacksFrom([upper, lower]),
    version: 3,
    width: 64,
  };
  const h = createLayerOperationHarness(document, { backend });
  const fill = (id: string, rect: Rect, color: (x: number) => string) => {
    const entry = h.layers.getOrCreateRect(id, rect);
    for (let x = 0; x < rect.width; x += 1) {
      entry.surface.ctx.fillStyle = color(x);
      entry.surface.ctx.fillRect(x, 0, 1, rect.height);
    }
    h.layers.publishPixels(id);
  };
  fill('upper', UPPER_RECT, (x) => `rgba(${40 + x * 8}, 90, 200, ${x % 3 === 0 ? 0.6 : 1})`);
  fill('lower', LOWER_RECT, (x) => `rgb(200, ${50 + x * 9}, 70)`);
  const display = createAdjustedSurfaceCache(backend);
  const contracts = {
    lower: getDocumentLayer(h.document(), 'lower')!,
    upper: getDocumentLayer(h.document(), 'upper')!,
  };
  /** What the canvas shows over {@link VIEW}: each layer's display copy (adjusted when it has a stack) through its transform. */
  const canvasPixels = (): number[] => {
    const surface = backend.createSurface(VIEW.width, VIEW.height);
    for (const contract of [...getDocumentLeaves(h.document())].reverse()) {
      const entry = h.layers.get(contract.id)!;
      const adjustments = contract.type === 'raster' ? contract.adjustments : undefined;
      const pixels = display.get(contract.id, entry, adjustments) ?? entry.surface;
      const { rotation, scaleX, scaleY, x, y } = contract.transform;
      const matrix = fromTRS({ x, y }, rotation, scaleX, scaleY);
      surface.ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - VIEW.x, matrix.f - VIEW.y);
      surface.ctx.drawImage(pixels.canvas, entry.rect.x, entry.rect.y);
    }
    return [...surface.ctx.getImageData(0, 0, VIEW.width, VIEW.height).data];
  };
  const rawPixels = (id: string): number[] => {
    const entry = h.layers.get(id)!;
    return [...entry.surface.ctx.getImageData(0, 0, entry.rect.width, entry.rect.height).data];
  };
  const controller = new MergeLayerController({
    backend,
    ctx: h.ctx,
    exportBaked: () => Promise.reject(new Error('merge down reads live caches')),
    hasExportableContent: (layerId) => (h.layers.get(layerId)?.rect.width ?? 0) > 0,
    isCacheReady: () => true,
    layers: h.layers,
    needsPixelPersistence: () => false,
    peekAdjustedSurface: (contract, entry) =>
      displayed && contract.type === 'raster' ? display.get(contract.id, entry, contract.adjustments) : null,
    publishSelectedLayerIds: () => undefined,
  });
  return { ...h, canvasPixels, contracts, controller, rawPixels };
};

describe('Merge Down with adjustment stacks', () => {
  describe.each([
    ['noise on the upper layer', { upperAdjustments: NOISE }],
    ['noise on the layer below', { lowerAdjustments: NOISE }],
    [
      'noise on a rotated, downscaled upper layer above a color-adjusted one',
      {
        lowerAdjustments: [{ id: 'invert', isEnabled: true, type: 'invert' }],
        upperAdjustments: NOISE,
        upperTransform: { rotation: Math.PI / 7, scaleX: 0.7, scaleY: 0.55, x: 9, y: 4 },
      },
    ],
  ] satisfies [string, Scene][])('%s', (_, scene) => {
    it.each([
      ['adjusting layer-local copies', false],
      ["drawing the display's current adjusted copies", true],
    ])('keeps the displayed pixels by %s; undo restores both raw layers and stacks', async (_label, displayed) => {
      const h = mergeScene(scene, displayed);
      const raw = { lower: h.rawPixels('lower'), upper: h.rawPixels('upper') };
      const shown = h.canvasPixels();

      expect(h.controller.mergeDown('upper')).toBe('merged');

      // The merged layer has no stack: its baked pixels show what both layers showed.
      expect(getDocumentLayer(h.document(), 'lower')).not.toHaveProperty('adjustments');
      expect(h.canvasPixels()).toEqual(shown);

      await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
      expect(getDocumentLayer(h.document(), 'upper')).toBe(h.contracts.upper);
      expect(getDocumentLayer(h.document(), 'lower')).toBe(h.contracts.lower);
      expect(h.rawPixels('upper')).toEqual(raw.upper);
      expect(h.rawPixels('lower')).toEqual(raw.lower);
    });
  });
});
