import type {
  CanvasDocumentContractV3,
  CanvasInpaintMaskLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';
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

  it.each([
    { dpr: 1, pan: { x: 0, y: 0 }, zoom: 1 },
    { dpr: 1, pan: { x: 7, y: -3 }, zoom: 1 },
    { dpr: 2, pan: { x: 5.25, y: 2.5 }, zoom: 1.5 },
    { dpr: 1, pan: { x: -11, y: 4 }, zoom: 3 },
    { dpr: 1, pan: { x: 3, y: 1 }, zoom: 0.5 },
  ])('picks the hatch line or gap the viewport shows at zoom $zoom, DPR $dpr', ({ dpr, pan, zoom }) => {
    const backend = createDomRasterBackend();
    const caches = createLayerCacheStore(backend);
    const blue = caches.getOrCreate('blue', 128, 128);
    blue.surface.ctx.fillStyle = '#0000ff';
    blue.surface.ctx.fillRect(0, 0, 128, 128);
    caches.publishPixels('blue');
    const coverage = caches.getOrCreate('mask', 128, 128);
    coverage.surface.ctx.fillStyle = '#000000';
    coverage.surface.ctx.fillRect(0, 0, 128, 128);
    caches.publishPixels('mask');
    const mask: CanvasInpaintMaskLayerContract = {
      blendMode: 'normal',
      id: 'mask',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
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
    const tile = createMaskPatternTile(backend, 'diagonal', '#e07575')!;
    const view: Mat2d = { a: zoom * dpr, b: 0, c: 0, d: zoom * dpr, e: pan.x * dpr, f: pan.y * dpr };
    const screen = backend.createSurface(96, 96);
    compositeDocument(screen, doc, caches, view, {
      backend,
      imageSmoothing: shouldSmoothAtZoom(zoom),
      maskPatternTile: () => tile,
      screenMask: { dpr, getScratch: (width, height) => backend.createSurface(width, height) },
    });
    const sampler = createColorSampler(backend);
    const providers = { maskPatterns: { screen: () => ({ dpr, view }), tile: () => tile } };
    const toDocument = invert(view)!;

    const seen = new Set<string>();
    for (let y = 24; y < 36; y += 1) {
      for (let x = 24; x < 36; x += 1) {
        const [r, g, b] = screen.ctx.getImageData(x, y, 1, 1).data;
        const docPoint = applyToPoint(toDocument, { x: x + 0.25, y: y + 0.75 });
        const sample = sampler.sample(doc, caches, docPoint, providers);
        expect([sample?.r, sample?.g, sample?.b], `device pixel ${x},${y}`).toEqual([r, g, b]);
        seen.add(r === 0 && g === 0 && b === 255 ? 'gap' : 'line');
      }
    }
    expect([...seen].sort()).toEqual(['gap', 'line']);
  });
});
