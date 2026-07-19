import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { TransformSession } from '@workbench/canvas-engine/engineStores';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { TransformEditingController } from './transformEditingController';

const document = {
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [
    {
      blendMode: 'normal',
      id: 'shape',
      isEnabled: true,
      isLocked: false,
      name: 'Shape',
      opacity: 1,
      source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    },
  ],
  width: 100,
} as CanvasDocumentContractV2;

describe('TransformEditingController', () => {
  it('owns preview state and commits parametric transforms through declared ports', () => {
    let session: TransformSession | null = null;
    const dispatch = vi.fn();
    const pushHistory = vi.fn();
    const setOverride = vi.fn();
    const controller = new TransformEditingController({
      backend: createTestStubRasterBackend(),
      canEdit: () => true,
      dispatch,
      endBurst: vi.fn(),
      getCache: () => null,
      getDocument: () => document,
      invalidate: vi.fn(),
      isGestureActive: () => false,
      pushHistory,
      replaceCache: vi.fn(),
      restoreCache: vi.fn(),
      session: { get: () => session, set: (value) => (session = value) },
      setOverride,
    });

    controller.begin('shape');
    controller.update({ rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 4 });
    controller.apply();

    expect(session).toBeNull();
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'shape', patch: { transform: expect.objectContaining({ x: 12, y: 4 }) } })
    );
    expect(pushHistory).toHaveBeenCalledOnce();
    expect(setOverride).toHaveBeenLastCalledWith('shape', null);
  });
});
