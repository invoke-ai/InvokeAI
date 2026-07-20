import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';

import { createDecodedBitmapPool } from '@workbench/canvas-engine/render/decodedBitmapPool';
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
  bitmapPool: createDecodedBitmapPool(),
  documentSize: { height: 200, width: 300 },
  resolver,
  store: createLayerCacheStore(backend),
});

describe('rasterizeSource — image', () => {
  it('gives the shared image resolver a cancellable pool-owned signal', async () => {
    const { backend } = createSpyBackend();
    const controller = new AbortController();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = { ...makeDeps(resolver, backend), signal: controller.signal };

    await rasterizeSource({ image: imageRef('signaled'), type: 'image' }, deps);

    const resolverSignal = resolver.mock.calls[0]?.[1];
    expect(resolverSignal).toBeInstanceOf(AbortSignal);
    expect(resolverSignal).not.toBe(controller.signal);
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
    expect(deps.bitmapPool?.byteSize()).toBe(0);
  });

  it('decodes via the resolver + backend and coalesces concurrent callers', async () => {
    const { backend, createImageBitmap } = createSpyBackend();
    const resolver = vi.fn<ImageResolver>(() => Promise.resolve(new Blob()));
    const deps = makeDeps(resolver, backend);
    const source: CanvasLayerSourceContract = { image: imageRef('cat'), type: 'image' };

    const [resultA, resultB] = await Promise.all([rasterizeSource(source, deps), rasterizeSource(source, deps)]);

    // Both active rasterizations share one decode; the final release closes it.
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(createImageBitmap).toHaveBeenCalledTimes(1);
    expect(deps.bitmapPool?.byteSize()).toBe(0);

    // Surface sized to the image ref; content rect at the origin.
    expect(resultA.surface.width).toBe(32);
    expect(resultA.surface.height).toBe(16);
    expect(resultA.rect).toEqual({ height: 16, width: 32, x: 0, y: 0 });
    expect(resultB.surface.width).toBe(32);
  });

  it('keeps a shared decode alive when the first of two consumers aborts', async () => {
    const { backend, createImageBitmap } = createSpyBackend();
    let resolveBlob!: (blob: Blob) => void;
    const resolver = vi.fn<ImageResolver>(
      (_imageName, signal) =>
        new Promise<Blob>((resolve, reject) => {
          resolveBlob = resolve;
          signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
        })
    );
    const shared = makeDeps(resolver, backend);
    const firstController = new AbortController();
    const secondController = new AbortController();
    const source: CanvasLayerSourceContract = { image: imageRef('shared'), type: 'image' };

    const first = rasterizeSource(source, { ...shared, signal: firstController.signal });
    const second = rasterizeSource(source, { ...shared, signal: secondController.signal });
    await Promise.resolve();
    firstController.abort();
    resolveBlob(new Blob());

    await expect(first).rejects.toBe(firstController.signal.reason);
    await expect(second).resolves.toMatchObject({ rect: { height: 16, width: 32, x: 0, y: 0 } });
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(createImageBitmap).toHaveBeenCalledTimes(1);
    expect(shared.bitmapPool?.byteSize()).toBe(0);
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
