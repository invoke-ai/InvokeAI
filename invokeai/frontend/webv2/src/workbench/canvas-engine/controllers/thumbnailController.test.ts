import type { CanvasDocumentContractV2 } from '@workbench/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { ThumbnailController } from './thumbnailController';

describe('ThumbnailController budget', () => {
  it('returns over-budget before rasterizing a missing thumbnail cache', async () => {
    const layer = {
      blendMode: 'normal' as const,
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { image: { height: 100, imageName: 'layer.png', width: 100 }, type: 'image' as const },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster' as const,
    };
    const document: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [layer],
      selectedLayerId: null,
      version: 2,
      width: 100,
    };
    const rasterize = vi.fn();
    const controller = new ThumbnailController({
      backend: createTestStubRasterBackend(),
      getActiveProjectId: () => 'p',
      getCheckerboard: vi.fn(),
      getDocument: () => document,
      getEntry: () => undefined,
      getMaskPattern: () => null,
      isDisposed: () => false,
      isSupportedSource: () => true,
      projectId: 'p',
      rasterize,
      reportError: vi.fn(),
      reserve: () => ({ availableBytes: 0, requestedBytes: 80_000, status: 'over-budget' }),
      setStatus: vi.fn(),
    });

    await expect(controller.request('layer')).resolves.toBe('over-budget');
    expect(rasterize).not.toHaveBeenCalled();
  });
});
