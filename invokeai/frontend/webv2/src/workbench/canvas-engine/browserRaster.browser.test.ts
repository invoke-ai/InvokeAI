import type {
  CanvasDocumentContractV3,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { executePsdExport, planPsdExport } from '@workbench/canvas-engine/export/psdExport';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { createColorSampler } from '@workbench/canvas-engine/render/colorSample';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import { rasterizeShapeSource } from '@workbench/canvas-engine/render/rasterizers/shapeRasterizer';
import { rasterizeTextSource } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';
import { describe, expect, it, vi } from 'vitest';

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

const documentWith = (layers: CanvasRasterLayerContractV2[]): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 8, width: 8, x: 0, y: 0 },
  height: 8,
  stacks: stacksFrom(layers),
  selectedLayerId: layers[0]?.id ?? null,
  version: 3,
  width: 8,
});

describe('real browser raster acceptance', () => {
  it('samples safely with a real zero-sized cached layer surface', () => {
    const backend = createDomRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('empty', 0, 0);

    expect(() =>
      createColorSampler(backend).sample(documentWith([rasterLayer('empty')]), caches, { x: 1, y: 1 })
    ).not.toThrow();
  });

  it('uses readback-optimized layer caches for repeated brush-history reads', () => {
    const warningSpy = vi.spyOn(console, 'warn');

    try {
      const backend = createDomRasterBackend();
      const surface = createLayerCacheStore(backend).getOrCreate('paint', 16, 16).surface;

      for (let index = 0; index < 16; index += 1) {
        surface.ctx.getImageData(0, 0, 16, 16);
      }

      expect(warningSpy.mock.calls.flat().join(' ')).not.toContain('willReadFrequently');
    } finally {
      warningSpy.mockRestore();
    }
  });

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
      { backend, clipRect: { height: 8, width: 4, x: 0, y: 0 } }
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

  it('undoes and redoes a real pixel patch', async () => {
    const backend = createDomRasterBackend();
    const surface = backend.createSurface(1, 1);
    const before = new ImageData(new Uint8ClampedArray([255, 0, 0, 255]), 1, 1);
    const after = new ImageData(new Uint8ClampedArray([0, 0, 255, 255]), 1, 1);
    const history = createHistory();
    surface.ctx.putImageData(after, 0, 0);
    const entry = createImagePatchEntry({
      after,
      apply: (_layerId, rect, pixels) => {
        surface.ctx.putImageData(pixels, rect.x, rect.y);
        return Promise.resolve();
      },
      before,
      label: 'Browser pixel edit',
      layerId: 'paint',
      rect: { height: 1, width: 1, x: 0, y: 0 },
    });
    history.admit(entry.bytes)!.publish(entry);

    await history.undo();
    expectPixel(surface, 0, 0, [255, 0, 0, 255]);
    await history.redo();
    expectPixel(surface, 0, 0, [0, 0, 255, 255]);
  });

  it('round-trips a small real PSD with a folder, layer pixels and a merged preview of the contributing leaves', async () => {
    const backend = createDomRasterBackend();
    const fill = (color: string) => {
      const surface = backend.createSurface(2, 2);
      surface.ctx.fillStyle = color;
      surface.ctx.fillRect(0, 0, 2, 2);
      return surface;
    };
    const surfaces = { blue: fill('#0000ff'), red: fill('#ff0000') };
    const leaf = (id: string, name: string) => ({
      blendMode: 'normal' as const,
      contentRect: { height: 2, width: 2, x: 0, y: 0 },
      id,
      isEnabled: true,
      name,
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    // A disabled folder above the red base: its blue leaf is exported but stays out of the preview.
    const plan = planPsdExport([
      { children: [leaf('blue', 'Blue')], id: 'g', isEnabled: false, name: 'Group', type: 'group' },
      leaf('red', 'Red'),
    ]);
    let bytes: ArrayBuffer | null = null;

    await executePsdExport(plan, 'acceptance.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: (id) =>
        Promise.resolve({
          rect: { height: 2, width: 2, x: 0, y: 0 },
          surface: surfaces[id as keyof typeof surfaces],
        }),
    });

    expect(bytes).not.toBeNull();
    const { readPsd } = await import('ag-psd');
    const parsed = readPsd(bytes!, { useImageData: true });
    expect(parsed.width).toBe(2);
    expect(parsed.height).toBe(2);
    expect(parsed.children?.map((child) => child.name)).toEqual(['Red', 'Group']);
    expect(parsed.children![1]).toMatchObject({ hidden: true });
    expect(parsed.children![1]!.children?.map((child) => child.name)).toEqual(['Blue']);
    expect(Array.from(parsed.imageData!.data.slice(0, 4))).toEqual([255, 0, 0, 255]);
    expect(Array.from(parsed.children![0]!.imageData!.data.slice(0, 4))).toEqual([255, 0, 0, 255]);
    expect(Array.from(parsed.children![1]!.children![0]!.imageData!.data.slice(0, 4))).toEqual([0, 0, 255, 255]);
  });

  it('writes a merged preview pixel-equivalent to compositing the contributing leaves with their opacity and blend', async () => {
    const backend = createDomRasterBackend();
    const fill = (color: string) => {
      const surface = backend.createSurface(2, 2);
      surface.ctx.fillStyle = color;
      surface.ctx.fillRect(0, 0, 2, 2);
      return surface;
    };
    const surfaces = { base: fill('#ff8040'), tint: fill('#4080ff') };
    const leaf = (id: string, name: string, opacity: number, blendMode: 'normal' | 'multiply') => ({
      blendMode,
      contentRect: { height: 2, width: 2, x: 0, y: 0 },
      id,
      isEnabled: true,
      name,
      opacity,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    // An enabled folder above the base: its tinted leaf contributes at half opacity, multiplied.
    const plan = planPsdExport([
      { children: [leaf('tint', 'Tint', 0.5, 'multiply')], id: 'g', isEnabled: true, name: 'Group', type: 'group' },
      leaf('base', 'Base', 1, 'normal'),
    ]);
    let bytes: ArrayBuffer | null = null;
    await executePsdExport(plan, 'preview.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: (id) =>
        Promise.resolve({ rect: { height: 2, width: 2, x: 0, y: 0 }, surface: surfaces[id as keyof typeof surfaces] }),
    });

    const reference = backend.createSurface(2, 2);
    reference.ctx.drawImage(surfaces.base.canvas, 0, 0);
    reference.ctx.globalAlpha = 0.5;
    reference.ctx.globalCompositeOperation = 'multiply';
    reference.ctx.drawImage(surfaces.tint.canvas, 0, 0);
    const expected = Array.from(reference.ctx.getImageData(0, 0, 2, 2).data);

    const { readPsd } = await import('ag-psd');
    const parsed = readPsd(bytes!, { useImageData: true });
    expect(Array.from(parsed.imageData!.data)).toEqual(expected);
    expect(parsed.children?.map((child) => child.name)).toEqual(['Base', 'Group']);
  });

  it('rasterizes triangle and star shapes with the documented geometry', async () => {
    const backend = createDomRasterBackend();
    const deps = {
      backend,
      documentSize: { height: 64, width: 64 },
      resolver: () => Promise.resolve(new Blob()),
      store: createLayerCacheStore(backend),
    };
    const alphaAt = (surface: { ctx: CanvasRenderingContext2D }, x: number, y: number) =>
      surface.ctx.getImageData(x, y, 1, 1).data[3]!;

    const triangle = await rasterizeShapeSource(
      { fill: '#ff0000', height: 64, kind: 'triangle', stroke: null, strokeWidth: 0, type: 'shape', width: 64 },
      deps
    );
    const tri = triangle.surface as { ctx: CanvasRenderingContext2D };
    expect(alphaAt(tri, 32, 32)).toBeGreaterThan(0);
    expect(alphaAt(tri, 32, 4)).toBeGreaterThan(0);
    expect(alphaAt(tri, 4, 62)).toBeGreaterThan(0);
    expect(alphaAt(tri, 4, 4)).toBe(0);
    expect(alphaAt(tri, 60, 4)).toBe(0);

    const star = await rasterizeShapeSource(
      { fill: '#00ff00', height: 64, kind: 'star', stroke: null, strokeWidth: 0, type: 'shape', width: 64 },
      deps
    );
    const st = star.surface as { ctx: CanvasRenderingContext2D };
    expect(alphaAt(st, 32, 32)).toBeGreaterThan(0);
    expect(alphaAt(st, 32, 4)).toBeGreaterThan(0);
    expect(alphaAt(st, 46, 32)).toBeGreaterThan(0);
    // The left spike reaches where a triangle would be empty, and the bottom
    // concave notch is empty where a triangle would be filled — these two pins
    // discriminate the star from every other kind.
    expect(alphaAt(st, 5, 22)).toBeGreaterThan(0);
    expect(alphaAt(st, 32, 60)).toBe(0);
    expect(alphaAt(st, 52, 32)).toBe(0);
    expect(alphaAt(st, 4, 4)).toBe(0);
  });

  it('writes folder opacity/blend natively and isolates the folder in the merged preview', async () => {
    const backend = createDomRasterBackend();
    const fill = (color: string) => {
      const surface = backend.createSurface(2, 2);
      surface.ctx.fillStyle = color;
      surface.ctx.fillRect(0, 0, 2, 2);
      return surface;
    };
    const surfaces = { base: fill('#0000ff'), red: fill('#ff0000'), white: fill('#ffffff') };
    const leaf = (id: string, name: string) => ({
      blendMode: 'normal' as const,
      contentRect: { height: 2, width: 2, x: 0, y: 0 },
      id,
      isEnabled: true,
      name,
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    // Opaque red over opaque white inside a 50% folder over a blue base. The
    // FOLDER composite is pure red, so the preview must be red 50% over blue —
    // per-leaf flattening would let white bleed through.
    const plan = planPsdExport([
      {
        blendMode: 'normal',
        children: [leaf('red', 'Red'), leaf('white', 'White')],
        id: 'g',
        isEnabled: true,
        name: 'Faded',
        opacity: 0.5,
        type: 'group',
      },
      leaf('base', 'Base'),
    ]);
    let bytes: ArrayBuffer | null = null;
    await executePsdExport(plan, 'folder-opacity.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: (id) =>
        Promise.resolve({ rect: { height: 2, width: 2, x: 0, y: 0 }, surface: surfaces[id as keyof typeof surfaces] }),
    });

    const reference = backend.createSurface(2, 2);
    reference.ctx.drawImage(surfaces.base.canvas, 0, 0);
    reference.ctx.globalAlpha = 0.5;
    reference.ctx.drawImage(surfaces.red.canvas, 0, 0);
    const expected = Array.from(reference.ctx.getImageData(0, 0, 2, 2).data);

    const { readPsd } = await import('ag-psd');
    const parsed = readPsd(bytes!, { useImageData: true });
    expect(Array.from(parsed.imageData!.data)).toEqual(expected);
    const folder = parsed.children!.find((child) => child.name === 'Faded')!;
    expect(folder.opacity).toBeCloseTo(0.5, 2);
    expect(folder.blendMode).toBe('normal');
  });
});
