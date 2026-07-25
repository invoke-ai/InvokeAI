import type { CanvasDocumentContractV2, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';

import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { sampleDocumentColor } from './colorSample';
import { compositeDocument } from './compositor';
import { createLayerCacheStore } from './layerCache';

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

const documentWith = (rasterLayer: CanvasRasterLayerContractV2, width = 8): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 8, width, x: 0, y: 0 },
  height: 8,
  layers: [rasterLayer],
  selectedLayerId: rasterLayer.id,
  version: 2,
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
    compositeDocument(viewport, doc, caches, IDENTITY);

    expect([...viewport.ctx.getImageData(20, 0, 1, 1).data]).toEqual(EXPECTED_PIXEL);
    expect(sampleDocumentColor(doc, caches, backend, { x: 20, y: 0 })).toEqual(EXPECTED_COLOR);
  });

  it('matches a visible paint cache whose local content rectangle is offset', () => {
    const backend = createDomRasterBackend();
    const caches = createLayerCacheStore(backend);
    const entry = caches.getOrCreateRect('far-layer', { height: 2, width: 2, x: 20, y: 0 });
    entry.surface.ctx.fillStyle = '#ef1234';
    entry.surface.ctx.fillRect(0, 0, 2, 2);
    caches.publishPixels('far-layer');
    const doc = documentWith(layer(0), 32);
    const viewport = backend.createSurface(32, 8);
    compositeDocument(viewport, doc, caches, IDENTITY);

    expect([...viewport.ctx.getImageData(20, 0, 1, 1).data]).toEqual(EXPECTED_PIXEL);
    expect(sampleDocumentColor(doc, caches, backend, { x: 20, y: 0 })).toEqual(EXPECTED_COLOR);
  });
});
