import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { MergeLayerController } from './mergeLayerController';

/** Coverage alpha of a mask's baked pixels, filled uniformly over `rect`. */
const coverage = (rect: Rect, alpha: number): RasterSurface => {
  const surface = createDomRasterBackend().createSurface(rect.width, rect.height);
  surface.ctx.fillStyle = `rgba(255, 255, 255, ${alpha / 255})`;
  surface.ctx.fillRect(0, 0, rect.width, rect.height);
  return surface;
};

describe('MergeLayerController in Chromium: merge masks', () => {
  it('unions coverage source-over, each mask at its own opacity', async () => {
    // The top mask is half-covered at full opacity; the one below is fully covered at half opacity.
    const sources: Record<string, { alpha: number; opacity: number; rect: Rect }> = {
      below: { alpha: 255, opacity: 0.5, rect: { height: 10, width: 10, x: 5, y: 0 } },
      top: { alpha: 128, opacity: 1, rect: { height: 10, width: 10, x: 0, y: 0 } },
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      selectedLayerId: 'top',
      stacks: stacksFrom(
        ['top', 'below'].map((id) =>
          layerContract(id, 'inpaint_mask', { opacity: sources[id]!.opacity } as Partial<CanvasLayerContract>)
        )
      ),
      version: 3,
      width: 100,
    };
    const h = createLayerOperationHarness(document);
    const dom = createDomRasterBackend();
    // Real raw caches, which undo snapshots copy.
    for (const [layerId, { alpha, rect }] of Object.entries(sources)) {
      h.layers.installReplacement({ layerId, rect, surface: coverage(rect, alpha) });
    }
    const created: RasterSurface[] = [];
    const controller = new MergeLayerController({
      backend: {
        ...dom,
        createSurface: (width, height, options) => {
          const surface = dom.createSurface(width, height, options);
          created.push(surface);
          return surface;
        },
      },
      ctx: h.ctx,
      exportBaked: (layerId) =>
        Promise.resolve({
          guard: { layer: getDocumentLayer(h.document(), layerId) } as never,
          rect: { ...sources[layerId]!.rect },
          release: () => undefined,
          status: 'ok' as const,
          surface: coverage(sources[layerId]!.rect, sources[layerId]!.alpha),
        }),
      hasExportableContent: () => true,
      isCacheReady: () => true,
      layers: h.layers,
      needsPixelPersistence: () => true,
      publishSelectedLayerIds: () => undefined,
    });

    await expect(controller.mergeInpaintMasks(['top', 'below'])).resolves.toMatchObject({ status: 'merged' });

    const merged = created.find((surface) => surface.width === 15 && surface.height === 10)!;
    const alphaAt = (x: number): number => merged.ctx.getImageData(x, 5, 1, 1).data[3]!;
    // Top only: its own coverage. Below only: half of full coverage. Both: 128 over 128 = 128 + 128 * (1 - 128/255).
    expect(alphaAt(2)).toBeCloseTo(128, -0.5);
    expect(alphaAt(12)).toBeCloseTo(128, -0.5);
    expect(alphaAt(7)).toBeCloseTo(128 + 128 * (1 - 128 / 255), -0.5);
  });
});
