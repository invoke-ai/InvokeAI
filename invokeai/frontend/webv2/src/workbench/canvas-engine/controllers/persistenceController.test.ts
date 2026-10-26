import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { BitmapStore, BitmapStoreTimers } from '@workbench/canvas-engine/document/bitmapStore';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';

import { createBitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { PersistenceController } from './persistenceController';

/** Debounce timers that never fire, so only a barrier flushes. */
const idleTimers: BitmapStoreTimers = { clearTimeout: () => undefined, setTimeout: () => 0 };

/** A real bitmap store over two paint layers whose uploads the test resolves. */
const createPersistence = () => {
  const surface = createTestStubRasterBackend().createSurface(4, 4);
  const sources = new Map<string, CanvasLayerSourceContract>([
    ['a', { bitmap: null, type: 'paint' }],
    ['b', { bitmap: null, type: 'paint' }],
  ]);
  const uploads: { content: string; resolve: (result: CanvasImageUploadResult) => void }[] = [];
  const store = createBitmapStore({
    dispatch: (action) => {
      if (action.type === 'updateCanvasLayerSource') {
        sources.set(action.id, action.source);
      }
      return true;
    },
    encodeSurface: () => Promise.resolve(new Blob([`pixels-${String(uploads.length)}`])),
    getLayerSource: (layerId) => sources.get(layerId) ?? null,
    getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
    hashBlob: (blob) => blob.text(),
    timers: idleTimers,
    uploadImage: (blob) =>
      blob.text().then(
        (text) =>
          new Promise<CanvasImageUploadResult>((resolve) => {
            uploads.push({ content: text, resolve });
          })
      ),
  });
  const controller = new PersistenceController(store);
  /** Settles every upload started so far. */
  const settleUploads = async (): Promise<void> => {
    await vi.waitFor(() => expect(uploads.length).toBeGreaterThan(0));
    for (const upload of uploads.splice(0)) {
      upload.resolve({ height: 4, imageName: `${upload.content}.png`, width: 4 });
    }
  };
  return { controller, settleUploads, sources, store, uploads };
};

const createStoreDouble = (): BitmapStore => ({
  discardLayer: vi.fn(),
  dispose: vi.fn(),
  flushPendingUploads: vi.fn(() => Promise.resolve()),
  hasPendingClear: vi.fn(() => false),
  hasPendingWork: vi.fn(() => false),
  isSelfEcho: vi.fn(() => false),
  markLayerDirty: vi.fn(),
  reset: vi.fn(),
  suspendLayer: vi.fn(() => vi.fn()),
});

describe('PersistenceController', () => {
  it('provides a flush barrier and idempotent disposal', async () => {
    const store = createStoreDouble();
    const controller = new PersistenceController(store);
    await controller.flush();
    expect(store.flushPendingUploads).toHaveBeenCalledOnce();
    controller.dispose();
    controller.dispose();
    expect(store.dispose).toHaveBeenCalledOnce();
  });

  describe('flushLayer', () => {
    it('reports a layer without unsaved pixels as clean', async () => {
      const p = createPersistence();

      await expect(p.controller.flushLayer('a')).resolves.toBe('clean');
      expect(p.uploads).toEqual([]);
    });

    it('persists only the named layer', async () => {
      const p = createPersistence();
      p.store.markLayerDirty('b');
      p.store.markLayerDirty('a');

      const flushed = p.controller.flushLayer('a');
      await p.settleUploads();

      await expect(flushed).resolves.toBe('flushed');
      expect(p.sources.get('a')).toMatchObject({ bitmap: { imageName: 'pixels-0.png' } });
      expect(p.store.hasPendingWork('a')).toBe(false);
      expect(p.store.hasPendingWork('b')).toBe(true);
      expect(p.sources.get('b')).toEqual({ bitmap: null, type: 'paint' });
    });

    it('stops waiting when aborted', async () => {
      const p = createPersistence();
      p.store.markLayerDirty('a');
      const abort = new AbortController();

      const flushed = p.controller.flushLayer('a', { signal: abort.signal });
      await vi.waitFor(() => expect(p.uploads).toHaveLength(1));
      abort.abort();

      await expect(flushed).resolves.toBe('aborted');
      await expect(p.controller.flushLayer('a', { signal: AbortSignal.abort() })).resolves.toBe('aborted');
    });

    it('stops waiting for a suspended layer when aborted, and flushes it once released', async () => {
      const p = createPersistence();
      p.store.markLayerDirty('a');
      const release = p.store.suspendLayer('a');
      const abort = new AbortController();

      const flushed = p.controller.flushLayer('a', { signal: abort.signal });
      abort.abort();
      await expect(flushed).resolves.toBe('aborted');

      release();
      const next = p.controller.flushLayer('a');
      await p.settleUploads();
      await expect(next).resolves.toBe('flushed');
    });
  });
});
