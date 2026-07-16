import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { ControlPixelController } from './controlPixelController';

describe('ControlPixelController', () => {
  it('can be instantiated with narrow fakes and rejects edits without a document', () => {
    const controller = new ControlPixelController({
      applyImagePatch: vi.fn(),
      backend: createTestStubRasterBackend(),
      bitmapStore: { discardLayer: vi.fn(), markLayerDirty: vi.fn(), suspendLayer: vi.fn() } as never,
      canEdit: () => true,
      deleteDerived: vi.fn(),
      dispatchReplacement: vi.fn(),
      endBurst: vi.fn(),
      getActiveProjectId: () => 'project-1',
      getDocument: () => null,
      getTransformSession: () => null,
      history: {} as never,
      installPrepared: vi.fn(),
      invalidate: vi.fn(),
      isCacheReady: () => false,
      isOperationIdle: () => true,
      layers: {} as never,
      notifyPainted: vi.fn(),
      preparePixels: vi.fn(),
      projectId: 'project-1',
      publishStroke: vi.fn(),
      setTransformOverride: vi.fn(),
    });

    expect(controller.begin('control-1')).toBeNull();
    expect(controller.isOpenFor(['control-1'])).toBe(false);
    expect(() => controller.dispose()).not.toThrow();
  });
});
