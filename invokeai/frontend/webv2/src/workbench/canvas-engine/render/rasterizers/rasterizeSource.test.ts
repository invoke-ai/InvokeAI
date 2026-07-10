import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import type { ImageResolver, RasterizeDeps } from './types';

import { rasterizeSource } from './index';

const imageRef = (imageName: string, width = 32, height = 16): CanvasImageRef => ({ height, imageName, width });

/** A backend whose `createImageBitmap` is a spy returning a closable fake bitmap. */
const createSpyBackend = () => {
  const stub = createTestStubRasterBackend();
  const createImageBitmap = vi.fn((source: ImageBitmapSource): Promise<ImageBitmap> => {
    void source;
    return Promise.resolve({ close: vi.fn(), height: 16, width: 32 } as unknown as ImageBitmap);
  });
  const backend: RasterBackend = {
    createImageBitmap,
    createSurface: stub.createSurface,
    encodeSurface: stub.encodeSurface,
  };
  return { backend, createImageBitmap, createSurface: stub.createSurface };
};

const makeDeps = (resolver: ImageResolver, backend: RasterBackend): RasterizeDeps => ({
  backend,
  documentSize: { height: 200, width: 300 },
  resolver,
  store: createLayerCacheStore(backend),
});

describe('rasterizeSource — image', () => {
  it('forwards the rasterization abort signal to the image resolver', async () => {
    const { backend } = createSpyBackend();
    const controller = new AbortController();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = { ...makeDeps(resolver, backend), signal: controller.signal };

    await rasterizeSource({ image: imageRef('signaled'), type: 'image' }, deps);

    expect(resolver).toHaveBeenCalledWith('signaled', controller.signal);
  });

  it('closes a decoded bitmap instead of caching it when cancellation lands during decode', async () => {
    const { backend } = createSpyBackend();
    let resolveDecoded!: (bitmap: ImageBitmap) => void;
    const decoded = new Promise<ImageBitmap>((resolve) => {
      resolveDecoded = resolve;
    });
    const bitmap = { close: vi.fn(), height: 16, width: 32 } as unknown as ImageBitmap;
    backend.createImageBitmap = vi.fn(() => decoded);
    const controller = new AbortController();
    const deps = { ...makeDeps(() => Promise.resolve(new Blob()), backend), signal: controller.signal };

    const rasterized = rasterizeSource({ image: imageRef('cancelled-decode'), type: 'image' }, deps);
    await Promise.resolve();
    controller.abort();
    resolveDecoded(bitmap);

    await expect(rasterized).rejects.toBe(controller.signal.reason);
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    expect(deps.store.getBitmap('cancelled-decode')).toBeUndefined();
  });

  it('decodes via the resolver + backend and caches the bitmap per image name', async () => {
    const { backend, createImageBitmap } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);
    const source: CanvasLayerSourceContract = { image: imageRef('cat'), type: 'image' };

    const resultA = await rasterizeSource(source, deps);
    const resultB = await rasterizeSource(source, deps);

    // Second call reuses the cached bitmap: no extra resolve/decode.
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(createImageBitmap).toHaveBeenCalledTimes(1);
    expect(deps.store.getBitmap('cat')).toBeDefined();

    // Surface sized to the image ref; content rect at the origin.
    expect(resultA.surface.width).toBe(32);
    expect(resultA.surface.height).toBe(16);
    expect(resultA.rect).toEqual({ height: 16, width: 32, x: 0, y: 0 });
    expect(resultB.surface.width).toBe(32);
  });

  it('draws the decoded bitmap onto the surface', async () => {
    const { backend } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);

    const { surface } = await rasterizeSource({ image: imageRef('dog'), type: 'image' }, deps);
    const ops = (surface as ReturnType<ReturnType<typeof createTestStubRasterBackend>['createSurface']>).callLog.map(
      (e) => e.op
    );
    expect(ops).toContain('clearRect');
    expect(ops).toContain('drawImage');
  });
});

describe('rasterizeSource — paint', () => {
  it('null bitmap produces an EMPTY (zero-rect) surface (no drawImage)', async () => {
    const { backend, createImageBitmap } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);

    const { rect, surface } = await rasterizeSource({ bitmap: null, type: 'paint' }, deps);

    // Content-sized: an empty paint layer holds no pixels — a zero rect.
    expect(rect).toEqual({ height: 0, width: 0, x: 0, y: 0 });
    expect(surface.width).toBe(0);
    expect(surface.height).toBe(0);
    expect(resolver).not.toHaveBeenCalled();
    expect(createImageBitmap).not.toHaveBeenCalled();
    const ops = (surface as ReturnType<ReturnType<typeof createTestStubRasterBackend>['createSurface']>).callLog.map(
      (e) => e.op
    );
    expect(ops).not.toContain('drawImage');
  });

  it('a paint bitmap is decoded and blitted onto a content-sized surface at its offset', async () => {
    const { backend, createImageBitmap } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);

    const { rect, surface } = await rasterizeSource(
      { bitmap: imageRef('brush'), offset: { x: 40, y: 25 }, type: 'paint' },
      deps
    );

    // Sized to the bitmap dims, placed at the persisted offset.
    expect(surface.width).toBe(32);
    expect(surface.height).toBe(16);
    expect(rect).toEqual({ height: 16, width: 32, x: 40, y: 25 });
    expect(createImageBitmap).toHaveBeenCalledTimes(1);
  });

  it('a paint bitmap without an offset defaults to origin (0,0) — legacy back-compat', async () => {
    const { backend } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);

    const { rect } = await rasterizeSource({ bitmap: imageRef('brush'), type: 'paint' }, deps);
    expect(rect).toEqual({ height: 16, width: 32, x: 0, y: 0 });
  });
});

describe('rasterizeSource — unimplemented sources', () => {
  it('throws for the deferred polygon shape kind', () => {
    const { backend } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);
    const source = {
      fill: '#000000',
      height: 10,
      kind: 'polygon',
      stroke: null,
      strokeWidth: 0,
      type: 'shape',
      width: 10,
    } as unknown as CanvasLayerSourceContract;
    expect(() => rasterizeSource(source, deps)).toThrow(/not implemented/i);
  });

  it('rasterizes shape, gradient, and text sources (no longer throwing)', async () => {
    const { backend } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);
    const shape = {
      fill: '#000000',
      height: 10,
      kind: 'rect',
      stroke: null,
      strokeWidth: 0,
      type: 'shape',
      width: 10,
    } as CanvasLayerSourceContract;
    const gradient = {
      angle: 0,
      kind: 'linear',
      stops: [{ color: '#000000', offset: 0 }],
      type: 'gradient',
    } as CanvasLayerSourceContract;
    const text = {
      align: 'left',
      color: '#000000',
      content: 'hi',
      fontFamily: 'Inter',
      fontSize: 20,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    } as CanvasLayerSourceContract;
    await expect(rasterizeSource(shape, deps)).resolves.toBeDefined();
    await expect(rasterizeSource(gradient, deps)).resolves.toBeDefined();
    await expect(rasterizeSource(text, deps)).resolves.toBeDefined();
  });
});
