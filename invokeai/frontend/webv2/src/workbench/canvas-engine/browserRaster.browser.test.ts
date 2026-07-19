import type {
  CanvasDocumentContractV2,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { executePsdExport, planPsdExport } from '@workbench/canvas-engine/export/psdExport';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import { rasterizeTextSource } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';
import { describe, expect, it } from 'vitest';

const IDENTITY: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };

const readPixel = (surface: RasterSurface, x: number, y: number): number[] => [
  ...surface.ctx.getImageData(x, y, 1, 1).data,
];

const expectPixel = (surface: RasterSurface, x: number, y: number, expected: number[], tolerance = 0): void => {
  const actual = readPixel(surface, x, y);
  expected.forEach((channel, index) => {
    expect(actual[index]).toBeGreaterThanOrEqual(channel - tolerance);
    expect(actual[index]).toBeLessThanOrEqual(channel + tolerance);
  });
};

const rasterLayer = (
  id: string,
  overrides: Partial<CanvasRasterLayerContractV2> = {}
): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 8, imageName: id, width: 8 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const documentWith = (layers: CanvasRasterLayerContractV2[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 8, width: 8, x: 0, y: 0 },
  height: 8,
  layers,
  selectedLayerId: layers[0]?.id ?? null,
  version: 2,
  width: 8,
});

describe('real browser raster acceptance', () => {
  it('composites transformed layers through clipping, alpha, and multiply blend mode', () => {
    const backend = createDomRasterBackend();
    const caches = createLayerCacheStore(backend);
    const bottom = caches.getOrCreate('bottom', 8, 8);
    bottom.surface.ctx.fillStyle = '#ff0000';
    bottom.surface.ctx.fillRect(0, 0, 8, 8);
    caches.publishPixels('bottom');

    const top = caches.getOrCreate('top', 4, 8);
    top.surface.ctx.fillStyle = '#0000ff';
    top.surface.ctx.fillRect(0, 0, 4, 8);
    caches.publishPixels('top');

    const target = backend.createSurface(8, 8);
    compositeDocument(
      target,
      documentWith([
        rasterLayer('top', {
          blendMode: 'multiply',
          opacity: 0.5,
          source: { image: { height: 8, imageName: 'top', width: 4 }, type: 'image' },
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 2, y: 0 },
        }),
        rasterLayer('bottom'),
      ]),
      caches,
      IDENTITY,
      { clipRect: { height: 8, width: 4, x: 0, y: 0 } }
    );

    expectPixel(target, 1, 1, [255, 0, 0, 255]);
    expectPixel(target, 3, 1, [127, 0, 0, 255], 2);
    expectPixel(target, 5, 1, [0, 0, 0, 0]);
  });

  it('uses browser text metrics and produces real glyph pixels', async () => {
    const backend = createDomRasterBackend();
    const source: Extract<CanvasLayerSourceContract, { type: 'text' }> = {
      align: 'left',
      color: '#ff0000',
      content: 'Invoke',
      fontFamily: 'sans-serif',
      fontSize: 24,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    };
    const measureSurface = backend.createSurface(1, 1);
    measureSurface.ctx.font = '400 24px sans-serif';
    const browserWidth = Math.ceil(measureSurface.ctx.measureText(source.content).width);

    const result = await rasterizeTextSource(source, {
      backend,
      documentSize: { height: 64, width: 64 },
      resolver: () => Promise.resolve(new Blob()),
      store: createLayerCacheStore(backend),
    });
    const pixels = result.surface.ctx.getImageData(0, 0, result.surface.width, result.surface.height).data;

    expect(result.rect.width).toBe(browserWidth);
    expect(result.rect.height).toBe(Math.ceil(24 * 1.2));
    expect(pixels.some((channel, index) => index % 4 === 3 && channel > 0)).toBe(true);
  });

  it('encodes a PNG blob and decodes it through createImageBitmap', async () => {
    const backend = createDomRasterBackend();
    const source = backend.createSurface(3, 2);
    source.ctx.fillStyle = '#00ff00';
    source.ctx.fillRect(0, 0, 3, 2);

    const blob = await backend.encodeSurface(source);
    const bitmap = await backend.createImageBitmap(blob);
    const decoded = backend.createSurface(3, 2);
    decoded.ctx.drawImage(bitmap, 0, 0);

    expect(blob.type).toBe('image/png');
    expect(blob.size).toBeGreaterThan(0);
    expect(bitmap.width).toBe(3);
    expect(bitmap.height).toBe(2);
    expectPixel(decoded, 1, 1, [0, 255, 0, 255]);
    bitmap.close();
  });

  it('undoes and redoes a real pixel patch', () => {
    const backend = createDomRasterBackend();
    const surface = backend.createSurface(1, 1);
    const before = new ImageData(new Uint8ClampedArray([255, 0, 0, 255]), 1, 1);
    const after = new ImageData(new Uint8ClampedArray([0, 0, 255, 255]), 1, 1);
    const history = createHistory();
    surface.ctx.putImageData(after, 0, 0);
    history.push(
      createImagePatchEntry({
        after,
        apply: (_layerId, rect, pixels) => surface.ctx.putImageData(pixels, rect.x, rect.y),
        before,
        label: 'Browser pixel edit',
        layerId: 'paint',
        rect: { height: 1, width: 1, x: 0, y: 0 },
      })
    );

    history.undo();
    expectPixel(surface, 0, 0, [255, 0, 0, 255]);
    history.redo();
    expectPixel(surface, 0, 0, [0, 0, 255, 255]);
  });

  it('round-trips a small real PSD with layer and composite pixels', async () => {
    const backend = createDomRasterBackend();
    const layerSurface = backend.createSurface(2, 2);
    layerSurface.ctx.fillStyle = '#ff0000';
    layerSurface.ctx.fillRect(0, 0, 2, 2);
    const plan = planPsdExport([
      {
        blendMode: 'normal',
        contentRect: { height: 2, width: 2, x: 0, y: 0 },
        id: 'red',
        isEnabled: true,
        name: 'Red',
        opacity: 1,
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      },
    ]);
    let bytes: ArrayBuffer | null = null;

    await executePsdExport(plan, 'acceptance.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: () =>
        Promise.resolve({
          rect: { height: 2, width: 2, x: 0, y: 0 },
          surface: layerSurface,
        }),
    });

    expect(bytes).not.toBeNull();
    const { readPsd } = await import('ag-psd');
    const parsed = readPsd(bytes!, { useImageData: true });
    expect(parsed.width).toBe(2);
    expect(parsed.height).toBe(2);
    expect(parsed.children?.map((child) => child.name)).toEqual(['Red']);
    expect(Array.from(parsed.imageData!.data.slice(0, 4))).toEqual([255, 0, 0, 255]);
    expect(Array.from(parsed.children![0]!.imageData!.data.slice(0, 4))).toEqual([255, 0, 0, 255]);
  });
});
