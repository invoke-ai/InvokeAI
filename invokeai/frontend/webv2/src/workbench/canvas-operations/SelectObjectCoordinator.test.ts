import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { LayerExportGuard } from '@workbench/canvas-engine/engine';

import { createCanvasOperationController } from '@workbench/canvas-operations/operationController';
import { createCanvasOperationStores } from '@workbench/canvas-operations/operationStores';
import { describe, expect, it, vi } from 'vitest';

import type { CanvasSelectObjectCoordinatorDeps } from './contracts';

import { createSelectObjectOperationCoordinator } from './SelectObjectCoordinator';

const layer: CanvasRasterLayerContractV2 = {
  blendMode: 'normal',
  id: 'source',
  isEnabled: true,
  isLocked: false,
  name: 'Source',
  opacity: 1,
  source: { fill: '#fff', height: 12, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 16 },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};
const guard: LayerExportGuard = { cacheVersion: 1, documentGeneration: 1, layer, layerId: layer.id, projectId: 'p1' };

const createCoordinator = (readAutoProcessPreference: () => boolean) => {
  const stores = createCanvasOperationStores();
  const deps: CanvasSelectObjectCoordinatorDeps = {
    captureGuard: () => guard,
    clearOtherOperation: vi.fn(),
    clearPreview: vi.fn(),
    commitGenerated: vi.fn(),
    commitMask: vi.fn(),
    controller: createCanvasOperationController({ isGuardCurrent: (candidate) => candidate === guard }),
    decodePreview: vi.fn(),
    exportSource: vi.fn(),
    invalidateOverlay: vi.fn(),
    isGuardCurrent: (candidate) => candidate === guard,
    isSamToolActive: () => true,
    prepareStart: () => ({
      guard,
      layerId: layer.id,
      layerName: layer.name,
      layerType: 'raster',
      sourceRect: { height: 12, width: 16, x: 0, y: 0 },
      status: 'ready',
    }),
    projectId: 'p1',
    publishPreview: vi.fn(),
    readAutoProcessPreference,
    replaceSelection: vi.fn(),
    replaceTemporaryRestoreTool: vi.fn(),
    runGraph: vi.fn(),
    selectLayer: vi.fn(),
    setSamTool: vi.fn(),
    setViewTool: vi.fn(),
    stores,
    uploadIntermediate: vi.fn(),
  };
  return { coordinator: createSelectObjectOperationCoordinator(deps), stores };
};

describe('createSelectObjectOperationCoordinator', () => {
  it('starts each session from the Auto Process preference as it stands at that start', () => {
    let preference = true;
    const { coordinator, stores } = createCoordinator(() => preference);

    expect(coordinator.start(layer.id)).toBe('started');
    expect(stores.samSession.get()?.autoProcess).toBe(true);

    coordinator.cancel();
    preference = false;

    expect(coordinator.start(layer.id)).toBe('started');
    expect(stores.samSession.get()?.autoProcess).toBe(false);
    coordinator.cancel();
  });
});
