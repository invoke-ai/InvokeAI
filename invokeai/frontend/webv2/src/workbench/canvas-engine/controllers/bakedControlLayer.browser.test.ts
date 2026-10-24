/** Real-canvas checks that copying or converting a raster layer to a control layer keeps its displayed pixels. */

import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { SpatialRunner } from '@workbench/canvas-engine/render/exactAdjustments';
import type { Rect } from '@workbench/canvas-engine/types';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { createDefaultControlAdapter } from '@workbench/controlAdapters';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CopyLayerController, type CopyLayerControllerOptions } from './copyLayerController';
import { LayerMutationController } from './layerMutationController';
import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { RasterExportController } from './rasterExportController';

const ADJUSTMENTS: CanvasAdjustmentsContract = [
  { amount: 90, id: 'clarity', isEnabled: true, size: 8, type: 'clarity' },
  { brightness: 0.1, contrast: 0.2, id: 'bc', isEnabled: true, type: 'brightness-contrast' },
  { amount: 50, id: 'dehaze', isEnabled: true, type: 'dehaze' },
];
const TRANSFORM = { rotation: Math.PI / 9, scaleX: 0.6, scaleY: 1.3, x: 11, y: 5 };
const CONTENT: Rect = { height: 30, width: 40, x: -6, y: -4 };
const CONTROL_MODEL = { base: 'sdxl', model: 'union-model' } as const;

const LAYER: CanvasRasterLayerContractV2 = {
  adjustments: ADJUSTMENTS,
  blendMode: 'multiply',
  id: 'layer',
  isEnabled: false,
  isLocked: false,
  name: 'Layer',
  opacity: 0.7,
  source: { bitmap: { height: 30, imageName: 'layer.png', width: 40 }, offset: { x: -6, y: -4 }, type: 'paint' },
  transform: TRANSFORM,
  type: 'raster',
};

const services: { dispose(): void }[] = [];
afterEach(() => {
  for (const service of services.splice(0)) {
    service.dispose();
  }
});

/** A rotated, scaled raster layer with a spatial stack, its controllers over a real export bake, and its display. */
const scene = (runSpatial?: SpatialRunner) => {
  const backend = createDomRasterBackend();
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 64, width: 64, x: 0, y: 0 },
    height: 64,
    selectedLayerId: 'layer',
    stacks: stacksFrom([LAYER]),
    version: 3,
    width: 64,
  };
  const h = createLayerOperationHarness(document, { backend });
  const entry = h.layers.getOrCreateRect('layer', CONTENT);
  for (let x = 0; x < CONTENT.width; x += 1) {
    entry.surface.ctx.fillStyle = `rgba(${60 + x * 3}, ${x % 5 === 0 ? 20 : 128}, ${200 - x * 3}, ${x % 7 ? 1 : 0.5})`;
    entry.surface.ctx.fillRect(x, 2, 1, 26);
  }
  h.layers.publishPixels('layer');
  const pixelsOf = (layerId: string): number[] => {
    const cached = h.layers.get(layerId)!;
    return [...cached.surface.ctx.getImageData(0, 0, cached.rect.width, cached.rect.height).data];
  };
  const raw = pixelsOf('layer');

  const memory = new RasterMemoryBudgetController({ budgetBytes: 64 * 1024 * 1024 });
  const service = createSpatialAdjustmentService({
    reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
  });
  services.push(service);
  const run: SpatialRunner = runSpatial ?? ((request) => service.run(request));
  const isGuardCurrent = (guard: { layer: unknown; layerId: string; cacheVersion: number }) =>
    getDocumentLayer(h.document(), guard.layerId) === guard.layer &&
    h.layers.get(guard.layerId)?.version === guard.cacheVersion;
  const exporter = new RasterExportController({
    backend,
    captureGuard: (layer, cached) => ({
      cacheVersion: cached.version,
      documentGeneration: 1,
      layer,
      layerId: layer.id,
      projectId: 'p',
    }),
    getDocument: () => h.document(),
    getOrStartRasterization: () => Promise.resolve('published'),
    isGuardCurrent,
    isRasterizing: () => false,
    isSupportedSource: () => true,
    layers: h.layers,
    pin: () => ({ release: () => undefined }),
    runSpatial: run,
  });
  const exportBaked: CopyLayerControllerOptions['exportBaked'] = async (layerId) => {
    const result = await exporter.baked(layerId, { includeDisabled: true });
    if (result.status === 'ok' || result.status === 'adjustment-refused') {
      return result;
    }
    return { status: result.status === 'aborted' ? 'not-ready' : result.status };
  };
  const copy = new CopyLayerController({
    backend,
    commitLayerCopy: () => {
      throw new Error('a layer with pixels is never copied as a plain clone');
    },
    ctx: h.ctx,
    exportBaked,
    getControlModel: () => CONTROL_MODEL,
    isGuardCurrent,
  });
  const unused = () => {
    throw new Error('not used by conversion');
  };
  const mutation = new LayerMutationController({
    backend,
    captureCache: (layer) => {
      const cached = h.layers.get(layer.id);
      if (!cached) {
        return null;
      }
      const pixels = backend.createSurface(cached.rect.width, cached.rect.height);
      pixels.ctx.drawImage(cached.surface.canvas, 0, 0);
      return { pixels, rect: { ...cached.rect } };
    },
    ctx: h.ctx,
    discardPersisted: () => undefined,
    exportBaked,
    getControlModel: () => CONTROL_MODEL,
    getDuplicateRasterPlan: unused,
    getSelectedLayerIds: unused,
    hasPendingPixelWork: () => false,
    isGuardCurrent,
    needsPixelPersistence: (layer) => 'source' in layer && layer.source.type === 'paint',
    pinDuplicateRasterSources: unused,
    prepareDuplicateRasterSource: unused,
    publishSelectedLayerIds: unused,
    sameContract: (candidate, layer) =>
      areJsonValuesStructurallyEqual(getDocumentLayer(candidate, layer.id) ?? null, layer),
    scheduleDuplicateRasterization: unused,
    trackDetached: unused,
  });

  /** The settled display: the layer's exact adjusted cache through its transform, over `rect`. */
  const displayed = async (rect: Rect): Promise<number[]> => {
    const display = createAdjustedSurfaceCache(backend, undefined, { spatial: { run } });
    await vi.waitFor(
      () => {
        display.get('layer', entry, ADJUSTMENTS);
        expect(display.peek('layer', entry, ADJUSTMENTS)).not.toBeNull();
      },
      { interval: 20, timeout: 10_000 }
    );
    const surface = backend.createSurface(rect.width, rect.height);
    const matrix = fromTRS({ x: TRANSFORM.x, y: TRANSFORM.y }, TRANSFORM.rotation, TRANSFORM.scaleX, TRANSFORM.scaleY);
    surface.ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - rect.x, matrix.f - rect.y);
    surface.ctx.drawImage(display.peek('layer', entry, ADJUSTMENTS)!.canvas, CONTENT.x, CONTENT.y);
    return [...surface.ctx.getImageData(0, 0, rect.width, rect.height).data];
  };
  return { copy, displayed, h, mutation, pixelsOf, raw };
};

const bakedControl = (id: string, name: string, rect: Rect) => ({
  adapter: createDefaultControlAdapter(CONTROL_MODEL.base, CONTROL_MODEL.model),
  blendMode: LAYER.blendMode,
  id,
  isEnabled: LAYER.isEnabled,
  isLocked: LAYER.isLocked,
  name,
  opacity: LAYER.opacity,
  source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

describe('copying a raster layer with a spatial stack to a control layer', () => {
  it('adds a control layer holding the settled display, evaluated in the spatial worker', async () => {
    const { copy, displayed, h, pixelsOf } = scene();

    const result = await copy.copy('layer', 'control');

    expect(result).toEqual({ layerId: 'new-1', status: 'copied' });
    const rect = h.layers.get('new-1')!.rect;
    expect(getDocumentLayer(h.document(), 'new-1')).toEqual(bakedControl('new-1', 'Layer copy', rect));
    expect(pixelsOf('new-1')).toEqual(await displayed(rect));
    expect(getDocumentLayer(h.document(), 'layer')).toEqual(LAYER);
  });

  it('refuses with the layer name when the stack cannot run, adding nothing', async () => {
    const { copy, h } = scene(() => Promise.resolve({ status: 'too-large' }));
    const before = h.document();

    expect(await copy.copy('layer', 'control')).toEqual({
      ownerId: 'layer',
      ownerName: 'Layer',
      reason: 'too-large',
      status: 'adjustment-refused',
    });
    expect(h.document()).toBe(before);
    expect(h.history.canUndo()).toBe(false);
  });
});

describe('converting a raster layer with a spatial stack to a control layer', () => {
  it('bakes the settled display into the control pixels; undo restores the stack and raw pixels', async () => {
    const { displayed, h, mutation, pixelsOf, raw } = scene();
    const live = getDocumentLayer(h.document(), 'layer')!;

    expect(await mutation.convertToControl('Convert to control', live)).toBe('converted');

    const rect = h.layers.get('layer')!.rect;
    const expected = await displayed(rect);
    expect(getDocumentLayer(h.document(), 'layer')).toEqual(bakedControl('layer', 'Layer', rect));
    expect(pixelsOf('layer')).toEqual(expected);
    expect(h.history.entries().past).toEqual(['Convert to control']);

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'layer')).toEqual(LAYER);
    expect(h.layers.get('layer')!.rect).toEqual(CONTENT);
    expect(pixelsOf('layer')).toEqual(raw);

    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'layer')).toEqual(bakedControl('layer', 'Layer', rect));
    expect(pixelsOf('layer')).toEqual(expected);
  });

  it('refuses with the layer name when the stack cannot run, leaving the raster layer', async () => {
    const { h, mutation, pixelsOf, raw } = scene(() => Promise.resolve({ status: 'over-budget' }));
    const before = h.document();

    expect(await mutation.convertToControl('Convert to control', getDocumentLayer(before, 'layer')!)).toEqual({
      ownerId: 'layer',
      ownerName: 'Layer',
      reason: 'over-budget',
      status: 'adjustment-refused',
    });
    expect(h.document()).toBe(before);
    expect(pixelsOf('layer')).toEqual(raw);
    expect(h.history.canUndo()).toBe(false);
  });

  it('lands nothing when the layer changed while its stack was evaluated', async () => {
    const { h, mutation } = scene();
    const live = getDocumentLayer(h.document(), 'layer')!;
    const converting = mutation.convertToControl('Convert to control', live);
    h.ctx.dispatch({ id: 'layer', patch: { opacity: 0.2 }, type: 'updateCanvasLayer' });

    expect(await converting).toBe('not-ready');
    expect(getDocumentLayer(h.document(), 'layer')).toMatchObject({ opacity: 0.2, type: 'raster' });
    expect(h.history.canUndo()).toBe(false);
  });
});
