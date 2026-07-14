import type { BitmapStore } from '@workbench/canvas-engine/document/bitmapStore';

import { describe, expect, it, vi } from 'vitest';

import { PersistenceController } from './persistenceController';

const createStore = (): BitmapStore => ({
  discardLayer: vi.fn(),
  dispose: vi.fn(),
  flushPendingUploads: vi.fn(() => Promise.resolve()),
  isSelfEcho: vi.fn(() => false),
  markLayerDirty: vi.fn(),
  reset: vi.fn(),
  suspendLayer: vi.fn(() => vi.fn()),
});

describe('PersistenceController', () => {
  it('provides a flush barrier and idempotent disposal', async () => {
    const store = createStore();
    const controller = new PersistenceController(store);
    await controller.flush();
    expect(store.flushPendingUploads).toHaveBeenCalledOnce();
    controller.dispose();
    controller.dispose();
    expect(store.dispose).toHaveBeenCalledOnce();
  });
});
