import type {
  CanvasDocumentContractV3,
  CanvasInpaintMaskLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { createColorSampler } from './colorSample';
import { compositeDocument, shouldSmoothAtZoom } from './compositor';
import { createLayerCacheStore } from './layerCache';
import { createMaskPatternTile } from './maskFill';

const IDENTITY = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 } as const;
const EXPECTED_COLOR = { a: 255, b: 52, g: 18, r: 239 } as const;
const EXPECTED_PIXEL = [239, 18, 52, 255];

const layer = (x: number): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id: 'far-layer',
  isEnabled: true,
  isLocked: false,
  name: 'Far layer',
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 0 },
  type: 'raster',
});

const documentWith = (rasterLayer: CanvasRasterLayerContractV2, width = 8): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 8, width, x: 0, y: 0 },
  height: 8,
  stacks: stacksFrom([rasterLayer]),
  selectedLayerId: rasterLayer.id,
  version: 3,
  width,
});

describe('color sampling with real browser rasterization', () => {
  it('matches a visible translated layer beyond the document rectangle', () => {
    const backend = createDomRasterBackend();
    const caches = createLayerCacheStore(backend);
    const entry = caches.getOrCreate('far-layer', 2, 2);
    entry.surface.ctx.fillStyle = '#ef1234';
    entry.surface.ctx.fillRect(0, 0, 2, 2);
    caches.publishPixels('far-layer');
    const doc = documentWith(layer(20));
    const viewport = backend.createSurface(24, 8);
    compositeDocument(viewport, doc, caches, IDENTITY, { backend });

    expect([...viewport.ctx.getImageData(20, 0, 1, 1).data]).toEqual(EXPECTED_PIXEL);
    expect(createColorSampler(backend).sample(doc, caches, { x: 20, y: 0 })).toEqual(EXPECTED_COLOR);
  });

  it('samples the square around a pixel with the pick at its center and empty canvas transparent', () => {
    const backend = createDomRasterBackend();
    const caches = createLayerCacheStore(backend);
    const entry = caches.getOrCreate('far-layer', 8, 8);
    entry.surface.ctx.fillStyle = '#ef1234';
    entry.surface.ctx.fillRect(0, 0, 4, 8);
    entry.surface.ctx.fillStyle = '#1234ef';
    entry.surface.ctx.fillRect(4, 0, 4, 8);
    caches.publishPixels('far-layer');
    const doc = documentWith(layer(0));
    const sampler = createColorSampler(backend);

    const area = sampler.sampleArea(doc, caches, { x: 3.5, y: 0.5 }, 5)!;

    expect(area.center).toEqual(sampler.sample(doc, caches, { x: 3.5, y: 0.5 }));
    expect(area.center).toEqual(EXPECTED_COLOR);
    const pixel = (x: number, y: number) => [...area.pixels.ctx.getImageData(x, y, 1, 1).data];
    // The area spans document x 1..5 and y -2..2: red up to x 3, blue from x 4, nothing above the layer.
    expect(pixel(2, 2)).toEqual(EXPECTED_PIXEL);
    expect(pixel(3, 2)).toEqual([18, 52, 239, 255]);
    expect(pixel(2, 0)).toEqual([0, 0, 0, 0]);
  });

  type MaskStyle = CanvasInpaintMaskLayerContract['mask']['fill']['style'];

  /** A coral patterned mask over blue, drawn as the viewport would; `paintMask` paints its coverage. */
  const hatchScene = (
    style: MaskStyle,
    { dpr, pan, zoom }: { dpr: number; pan: { x: number; y: number }; zoom: number },
    paintMask: (ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D) => void = (ctx) =>
      ctx.fillRect(0, 0, 128, 128)
  ) => {
    const backend = createDomRasterBackend();
    const caches = createLayerCacheStore(backend);
    const blue = caches.getOrCreate('blue', 128, 128);
    blue.surface.ctx.fillStyle = '#0000ff';
    blue.surface.ctx.fillRect(0, 0, 128, 128);
    caches.publishPixels('blue');
    const coverage = caches.getOrCreate('mask', 128, 128);
    coverage.surface.ctx.fillStyle = '#000000';
    paintMask(coverage.surface.ctx);
    caches.publishPixels('mask');
    const mask: CanvasInpaintMaskLayerContract = {
      blendMode: 'normal',
      id: 'mask',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#e07575', style } },
      name: 'Mask',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    };
    const doc: CanvasDocumentContractV3 = {
      ...documentWith({ ...layer(0), id: 'blue' }, 128),
      height: 128,
      stacks: stacksFrom([mask, { ...layer(0), id: 'blue' }]),
    };
    const tile = createMaskPatternTile(backend, style, '#e07575')!;
    const view: Mat2d = { a: zoom * dpr, b: 0, c: 0, d: zoom * dpr, e: pan.x * dpr, f: pan.y * dpr };
    const smoothing = shouldSmoothAtZoom(zoom);
    const screen = backend.createSurface(96, 96);
    compositeDocument(screen, doc, caches, view, {
      backend,
      imageSmoothing: smoothing,
      maskPatternTile: () => tile,
      screenMask: { dpr, getScratch: (width, height) => backend.createSurface(width, height) },
    });
    const providers = { maskPatterns: { screen: () => ({ dpr, smoothing, view }), tile: () => tile } };
    const sampler = createColorSampler(backend);
    // The viewport's own screen-to-document conversion, whose round trip can miss the device pixel.
    const toDocument = (cssPoint: { x: number; y: number }) => ({
      x: (cssPoint.x - pan.x) / zoom,
      y: (cssPoint.y - pan.y) / zoom,
    });
    const pick = (cssPoint: { x: number; y: number }) =>
      sampler.sample(doc, caches, toDocument(cssPoint), providers, cssPoint);
    const shown = (x: number, y: number) => [...screen.ctx.getImageData(x, y, 1, 1).data];
    return { caches, doc, pick, providers, sampler, shown, toDocument };
  };

  it.each<{ dpr: number; pan: { x: number; y: number }; zoom: number; style?: MaskStyle }>([
    { dpr: 1, pan: { x: 0, y: 0 }, zoom: 1 },
    { dpr: 1, pan: { x: 7, y: -3 }, zoom: 1 },
    { dpr: 2, pan: { x: 5.25, y: 2.5 }, zoom: 1.5 },
    { dpr: 1, pan: { x: -11, y: 4 }, zoom: 3 },
    { dpr: 1, pan: { x: 3, y: 1 }, zoom: 0.5 },
    { dpr: 1, pan: { x: 0, y: 0 }, zoom: 0.7, style: 'vertical' },
  ])(
    'picks the hatch line or gap the viewport shows at zoom $zoom, DPR $dpr',
    ({ style = 'diagonal', ...viewport }) => {
      const { pick, shown } = hatchScene(style, viewport);
      const seen = new Set<string>();
      for (let y = 24; y < 36; y += 1) {
        for (let x = 24; x < 36; x += 1) {
          const [r, g, b] = shown(x, y);
          // Integer pointer coordinates sit on device pixel edges, where a document round trip can land on a neighbor.
          for (const offset of [0, 0.25, 0.75]) {
            const sample = pick({ x: (x + offset) / viewport.dpr, y: (y + offset) / viewport.dpr });
            expect([sample?.r, sample?.g, sample?.b], `device pixel ${x},${y} + ${offset}`).toEqual([r, g, b]);
          }
          seen.add(r === 0 && g === 0 && b === 255 ? 'gap' : 'line');
        }
      }
      expect([...seen].sort()).toEqual(['gap', 'line']);
    }
  );

  it('picks one color across a screen pixel that blends a thin hatched stroke at 50% zoom', () => {
    const { pick, shown } = hatchScene('horizontal', { dpr: 1, pan: { x: 0, y: 0 }, zoom: 0.5 }, (ctx) =>
      ctx.fillRect(40, 0, 1, 128)
    );
    let blended = 0;
    for (let y = 0; y < 48; y += 1) {
      const [r, g, b] = shown(20, y);
      if (r !== 0 || g !== 0 || b !== 255) {
        blended += 1;
        expect(r).toBeLessThan(0xe0);
      }
      for (const x of [20.1, 20.9]) {
        const sample = pick({ x, y: y + 0.5 });
        expect([sample?.r, sample?.g, sample?.b], `screen ${x},${y}`).toEqual([r, g, b]);
      }
    }
    expect(blended).toBeGreaterThan(0);
  });

  it('samples the raster beneath a visible patterned mask when mask stacks are omitted', () => {
    const { caches, doc, providers, sampler, shown, toDocument } = hatchScene('vertical', {
      dpr: 1,
      pan: { x: 0, y: 0 },
      zoom: 2,
    });
    const point = { x: 32.5, y: 32.5 };
    expect(shown(32, 32)).not.toEqual([0, 0, 255, 255]);

    const imageProviders = {
      ...providers,
      showOverlayStacks: { control: true, inpaint_mask: false, regional_guidance: false, vector: true },
    };
    const expected = { r: 0, g: 0, b: 255, a: 255 };
    expect(sampler.sample(doc, caches, toDocument(point), imageProviders, point)).toEqual(expected);
    const area = sampler.sampleArea(doc, caches, toDocument(point), 3, imageProviders, point)!;
    expect(area.center).toEqual(expected);
    expect([...area.pixels.ctx.getImageData(0, 0, 1, 1).data]).toEqual([0, 0, 255, 255]);
  });

  it('fills each loupe cell with the screen pixel where its document pixel is drawn at 4x zoom', () => {
    const zoom = 4;
    const { caches, doc, providers, sampler, shown, toDocument } = hatchScene(
      'diagonal',
      { dpr: 1, pan: { x: -40, y: -40 }, zoom },
      (ctx) => ctx.fillRect(10, 10, 14, 14)
    );
    const pointer = { x: 41.3, y: 42.6 };
    const size = 7;
    const half = (size - 1) / 2;
    const area = sampler.sampleArea(doc, caches, toDocument(pointer), size, providers, pointer)!;

    const seen = new Set<string>();
    for (let row = 0; row < size; row += 1) {
      for (let column = 0; column < size; column += 1) {
        const x = Math.floor(pointer.x) + (column - half) * zoom;
        const y = Math.floor(pointer.y) + (row - half) * zoom;
        const expected = shown(x, y);
        expect([...area.pixels.ctx.getImageData(column, row, 1, 1).data], `cell ${column},${row}`).toEqual(expected);
        seen.add(expected.join());
      }
    }
    expect(seen.size).toBeGreaterThan(1);
  });
});
