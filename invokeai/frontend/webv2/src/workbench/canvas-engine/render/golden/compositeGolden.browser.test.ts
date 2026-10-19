import type {
  CanvasBlendMode,
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasMaskFillContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import {
  compositeDocument,
  createCheckerboardTile,
  type CompositeOptions,
} from '@workbench/canvas-engine/render/compositor';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createMaskPatternTile } from '@workbench/canvas-engine/render/maskFill';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { planBaseRasterComposite, renderRasterComposite } from '@workbench/canvas-engine/render/rasterComposite';
import { describe, expect, it } from 'vitest';
import { commands } from 'vitest/browser';

import { EXACT, expectGolden, goldenArtifactPath, INTERPOLATED } from './goldenHarness';

const WIDTH = 160;
const HEIGHT = 120;
const BBOX: Rect = { height: 90, width: 120, x: 20, y: 15 };

const transform = (overrides: Partial<CanvasLayerContract['transform']> = {}): CanvasLayerContract['transform'] => ({
  rotation: 0,
  scaleX: 1,
  scaleY: 1,
  x: 0,
  y: 0,
  ...overrides,
});

const raster = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
  transform: transform(),
  type: 'raster',
  ...overrides,
});

const control = (id: string, overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: transform(),
  type: 'control',
  withTransparencyEffect: false,
  ...overrides,
});

const inpaintMask = (id: string, fill: CanvasMaskFillContract): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill },
  name: id,
  opacity: 1,
  transform: transform(),
  type: 'inpaint_mask',
});

const doc = (layers: CanvasLayerContract[]): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: BBOX,
  height: HEIGHT,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
  width: WIDTH,
});

type Paint = (ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, rect: Rect) => void;

const solid =
  (color: string): Paint =>
  (ctx, rect) => {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, rect.width, rect.height);
  };

const stripes: Paint = (ctx, rect) => {
  ctx.fillStyle = '#f2c14e';
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#2d6a4f';
  for (let x = 0; x < rect.width; x += 16) {
    ctx.fillRect(x, 0, 8, rect.height);
  }
};

const scene = (layers: CanvasLayerContract[], paint: Record<string, { rect: Rect; paint: Paint }>) => {
  const backend = createDomRasterBackend();
  const caches = createLayerCacheStore(backend);
  for (const layer of layers) {
    const entry = paint[layer.id]!;
    caches.growToRect(layer.id, entry.rect);
    const surface = caches.get(layer.id)!.surface;
    surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
    entry.paint(surface.ctx, entry.rect);
    caches.publishPixels(layer.id);
  }
  return { backend, caches };
};

const render = (
  layers: CanvasLayerContract[],
  paint: Parameters<typeof scene>[1],
  options: Partial<CompositeOptions> = {}
) => {
  const { backend, caches } = scene(layers, paint);
  const target = backend.createSurface(WIDTH, HEIGHT);
  compositeDocument(target, doc(layers), caches, identity(), {
    backend,
    checkerboardTile: createCheckerboardTile(backend),
    ...options,
  });
  return target;
};

const BASE = { paint: stripes, rect: { height: 70, width: 100, x: 30, y: 25 } };
const OVERLAY = { paint: solid('#4f8ff2'), rect: { height: 50, width: 60, x: 70, y: 45 } };

describe('compositor goldens', () => {
  it('composites two rasters over the checkerboard', async () => {
    const layers = [raster('overlay'), raster('base')];
    await expectGolden('composition-normal', render(layers, { base: BASE, overlay: OVERLAY }));
  });

  for (const blendMode of ['multiply', 'screen', 'difference'] satisfies CanvasBlendMode[]) {
    it(`blends the upper raster with ${blendMode}`, async () => {
      const layers = [raster('overlay', { blendMode }), raster('base')];
      await expectGolden(`blend-${blendMode}`, render(layers, { base: BASE, overlay: OVERLAY }));
    });
  }

  it('applies layer opacity', async () => {
    const layers = [raster('overlay', { opacity: 0.5 }), raster('base')];
    await expectGolden('opacity-half', render(layers, { base: BASE, overlay: OVERLAY }));
  });

  it('rotates a layer through its transform', async () => {
    const layers = [
      raster('overlay', { transform: transform({ rotation: Math.PI / 6, x: 10, y: -6 }) }),
      raster('base'),
    ];
    await expectGolden('rotation-30', render(layers, { base: BASE, overlay: OVERLAY }), { tolerance: INTERPOLATED });
  });

  it('clips document layers to the clip rect but not the checkerboard', async () => {
    const layers = [raster('overlay'), raster('base')];
    await expectGolden(
      'clip-rect',
      render(layers, { base: BASE, overlay: OVERLAY }, { clipRect: { height: 60, width: 80, x: 40, y: 30 } })
    );
  });

  for (const style of ['solid', 'grid'] satisfies CanvasMaskFillContract['style'][]) {
    it(`colorizes an inpaint mask with a ${style} fill`, async () => {
      const layers = [inpaintMask('mask', { color: '#e07575', style }), raster('base')];
      const { backend, caches } = scene(layers, {
        base: BASE,
        mask: { paint: solid('#000000'), rect: { height: 40, width: 50, x: 50, y: 35 } },
      });
      const target = backend.createSurface(WIDTH, HEIGHT);
      compositeDocument(target, doc(layers), caches, identity(), {
        backend,
        checkerboardTile: createCheckerboardTile(backend),
        maskPatternTile: (fillStyle, color) => createMaskPatternTile(backend, fillStyle as typeof style, color),
      });
      await expectGolden(`mask-${style}`, target);
    });
  }

  it('skips a hidden control layer normally but draws it while isolated', async () => {
    const layers = [control('control', { isHidden: true }), raster('base')];
    const paint = { base: BASE, control: { paint: solid('#8a2be2'), rect: { height: 30, width: 90, x: 35, y: 30 } } };
    await expectGolden('hidden-overlay', render(layers, paint));
    await expectGolden('hidden-overlay-isolated', render(layers, paint, { isolationLayerId: 'control' }));
  });

  it('renders the generation base from enabled raster layers only', async () => {
    const layers = [
      control('control'),
      raster('overlay', { opacity: 0.75 }),
      raster('disabled', { isEnabled: false }),
      raster('base'),
    ];
    const { backend, caches } = scene(layers, {
      base: BASE,
      control: { paint: solid('#8a2be2'), rect: { height: 30, width: 90, x: 35, y: 30 } },
      disabled: { paint: solid('#ff0000'), rect: { height: 90, width: 120, x: 20, y: 15 } },
      overlay: OVERLAY,
    });
    const surface = await renderRasterComposite(planBaseRasterComposite(doc(layers), BBOX), {
      backend,
      getLayerSurface: (layerId) => {
        const entry = caches.get(layerId)!;
        return Promise.resolve({ rect: entry.rect, release: () => undefined, surface: entry.surface });
      },
    });
    await expectGolden('generation-base', surface, { tolerance: EXACT });
  });
});

describe('golden harness', () => {
  it.skipIf(__CANVAS_GOLDEN_UPDATE__)('refuses a frame without a reviewed baseline', async () => {
    const target = render([raster('base')], { base: BASE });
    await expect(expectGolden('unreviewed-frame', target)).rejects.toThrow(/Missing golden "unreviewed-frame"/);
  });

  it.skipIf(__CANVAS_GOLDEN_UPDATE__)(
    'reports a differing frame and writes expected, actual and diff artifacts',
    async () => {
      const altered = render([raster('overlay'), raster('base')], {
        base: BASE,
        overlay: { ...OVERLAY, paint: solid('#f24f4f') },
      });
      await expect(
        expectGolden('composition-normal', altered, { artifactName: 'negative-composition' })
      ).rejects.toThrow(/Golden "composition-normal" differs/);
      for (const kind of ['expected', 'actual', 'diff'] as const) {
        const png = await commands.readFile(goldenArtifactPath('negative-composition', kind), 'base64');
        expect(png.length).toBeGreaterThan(0);
      }
    }
  );

  it.skipIf(__CANVAS_GOLDEN_UPDATE__)('accepts interpolation noise only within the declared tolerance', async () => {
    const shifted = render(
      [raster('overlay', { transform: transform({ rotation: Math.PI / 6 + 0.002, x: 10, y: -6 }) }), raster('base')],
      { base: BASE, overlay: OVERLAY }
    );
    await expect(
      expectGolden('rotation-30', shifted, { artifactName: 'negative-rotation', tolerance: INTERPOLATED })
    ).rejects.toThrow(/differs/);
  });
});
