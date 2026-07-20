import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';

import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { SelectionPixelController } from './selectionPixelController';

describe('SelectionPixelController', () => {
  it('owns raster fill persistence and history through declared ports', () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreateRect('paint', { height: 2, width: 2, x: 0, y: 0 });
    const mask = backend.createSurface(2, 2);
    mask.ctx.fillStyle = '#fff';
    mask.ctx.fillRect(0, 0, 2, 2);
    const selection = {
      bounds: () => ({ height: 2, width: 2, x: 0, y: 0 }),
      mask: () => ({ rect: { height: 2, width: 2, x: 0, y: 0 }, surface: mask }),
    } as unknown as SelectionState;
    const document = {
      layers: [
        {
          id: 'paint',
          isEnabled: true,
          isLocked: false,
          source: { type: 'paint' },
          type: 'raster',
        },
      ],
      selectedLayerId: 'paint',
    } as CanvasDocumentContractV2;
    const history = createHistory();
    const markDirty = vi.fn();
    const notifyPainted = vi.fn();
    const controller = new SelectionPixelController({
      applyImagePatch: vi.fn(),
      backend,
      beginControlEdit: () => null,
      canEdit: () => true,
      deleteDerived: vi.fn(),
      endBurst: vi.fn(),
      getDocument: () => document,
      getFillColor: () => '#f00',
      history,
      invalidateLayer: vi.fn(),
      isGestureActive: () => false,
      layers,
      markDirty,
      notifyPainted,
      selection,
    });

    controller.run('fill');

    expect(markDirty).toHaveBeenCalledWith('paint');
    expect(notifyPainted).toHaveBeenCalledWith('paint');
    expect(history.canUndo()).toBe(true);
  });
});
