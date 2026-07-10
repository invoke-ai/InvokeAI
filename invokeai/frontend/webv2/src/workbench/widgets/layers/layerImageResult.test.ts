import type { LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { describe, expect, it, vi } from 'vitest';

import type { SelectObjectRouteDeps, SelectObjectRunnerDeps } from './layerImageResult';

import {
  createDefaultSelectObjectOptions,
  isSelectObjectPromptValid,
  routeSelectObjectResult,
  runSelectObject,
} from './layerImageResult';

const layer: CanvasRasterLayerContractV2 = {
  blendMode: 'normal',
  id: 'source',
  isEnabled: true,
  isLocked: false,
  name: 'Source',
  opacity: 1,
  source: { fill: '#fff', height: 12, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 16 },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: -5, y: 7 },
  type: 'raster',
};
const guard: LayerExportGuard = {
  cacheVersion: 3,
  documentGeneration: 4,
  layer,
  layerId: layer.id,
  projectId: 'p1',
};
const rect = { height: 12, width: 16, x: -5, y: 7 };
const blob = new Blob(['source'], { type: 'image/png' });
const ready = {
  guard,
  image: { height: 12, imageName: 'sam-output.png', width: 16 },
  rect,
  status: 'ready' as const,
};

const createDeferred = <T>(): { promise: Promise<T>; resolve(value: T): void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const createRunnerDeps = (overrides: Partial<SelectObjectRunnerDeps> = {}): SelectObjectRunnerDeps => ({
  exportLayer: vi.fn(() => Promise.resolve({ blob, guard, rect, status: 'ok' as const })),
  runGraph: vi.fn(() => Promise.resolve({ imageName: 'sam-output.png', origin: 'webv2:util:test' })),
  uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'sam-input.png' })),
  ...overrides,
});

describe('select-object defaults and validation', () => {
  it('defaults to SAM 2 Large, no polygon refinement, and a transient selection', () => {
    expect(createDefaultSelectObjectOptions()).toEqual({
      applyPolygonRefinement: false,
      model: 'segment-anything-2-large',
      prompt: '',
      target: 'selection',
    });
  });

  it('requires a non-blank object prompt', () => {
    expect(isSelectObjectPromptValid('   ')).toBe(false);
    expect(isSelectObjectPromptValid('  cat  ')).toBe(true);
  });
});

describe('runSelectObject', () => {
  it('exports, uploads, builds, and runs the exact SAM utility graph in order', async () => {
    const calls: string[] = [];
    const controller = new AbortController();
    const deps = createRunnerDeps({
      exportLayer: vi.fn((layerId) => {
        expect(layerId).toBe('source');
        calls.push('export');
        return Promise.resolve({ blob, guard, rect, status: 'ok' as const });
      }),
      runGraph: vi.fn(({ graph, outputNodeId, signal }) => {
        calls.push('run');
        expect(outputNodeId).toBe('sam-output');
        expect(signal).toBe(controller.signal);
        expect(graph.nodes['sam-detect']).toMatchObject({
          detection_threshold: 0.3,
          prompt: 'cat',
          type: 'grounding_dino',
        });
        expect(graph.nodes['sam-segment']).toMatchObject({
          apply_polygon_refinement: true,
          model: 'segment-anything-huge',
          type: 'segment_anything',
        });
        return Promise.resolve({ imageName: 'sam-output.png', origin: 'webv2:util:test' });
      }),
      uploadIntermediate: vi.fn((receivedBlob, signal) => {
        expect(receivedBlob).toBe(blob);
        expect(signal).toBe(controller.signal);
        calls.push('upload');
        return Promise.resolve({ imageName: 'sam-input.png' });
      }),
    });

    await expect(
      runSelectObject({
        applyPolygonRefinement: true,
        deps,
        layerId: 'source',
        model: 'segment-anything-huge',
        prompt: ' cat ',
        signal: controller.signal,
      })
    ).resolves.toEqual(ready);
    expect(calls).toEqual(['export', 'upload', 'run']);
  });

  it.each(['missing', 'disabled', 'unsupported', 'empty', 'not-ready'] as const)(
    'forwards an export %s status without uploading',
    async (status) => {
      const deps = createRunnerDeps({ exportLayer: vi.fn(() => Promise.resolve({ status })) });

      await expect(
        runSelectObject({
          applyPolygonRefinement: false,
          deps,
          layerId: 'source',
          model: 'segment-anything-2-large',
          prompt: 'cat',
        })
      ).resolves.toEqual({ status });
      expect(deps.uploadIntermediate).not.toHaveBeenCalled();
      expect(deps.runGraph).not.toHaveBeenCalled();
    }
  );

  it.each(['before export', 'after export', 'after upload', 'after graph'] as const)(
    'returns aborted $label and starts no later boundary',
    async (label) => {
      const controller = new AbortController();
      if (label === 'before export') {
        controller.abort();
      }
      const deps = createRunnerDeps({
        exportLayer: vi.fn(() => {
          if (label === 'after export') {
            controller.abort();
          }
          return Promise.resolve({ blob, guard, rect, status: 'ok' as const });
        }),
        runGraph: vi.fn(() => {
          if (label === 'after graph') {
            controller.abort();
          }
          return Promise.resolve({ imageName: 'sam-output.png', origin: 'webv2:util:test' });
        }),
        uploadIntermediate: vi.fn(() => {
          if (label === 'after upload') {
            controller.abort();
          }
          return Promise.resolve({ imageName: 'sam-input.png' });
        }),
      });

      await expect(
        runSelectObject({
          applyPolygonRefinement: false,
          deps,
          layerId: 'source',
          model: 'segment-anything-2-large',
          prompt: 'cat',
          signal: controller.signal,
        })
      ).resolves.toEqual({ status: 'aborted' });
      expect(deps.exportLayer).toHaveBeenCalledTimes(label === 'before export' ? 0 : 1);
      expect(deps.uploadIntermediate).toHaveBeenCalledTimes(
        label === 'before export' || label === 'after export' ? 0 : 1
      );
      expect(deps.runGraph).toHaveBeenCalledTimes(label === 'after graph' ? 1 : 0);
    }
  );

  it('maps AbortError to aborted and unexpected errors to failed', async () => {
    const aborted = createRunnerDeps({
      uploadIntermediate: vi.fn(() => Promise.reject(new DOMException('', 'AbortError'))),
    });
    const failed = createRunnerDeps({ runGraph: vi.fn(() => Promise.reject(new Error('SAM graph failed'))) });
    const options = {
      applyPolygonRefinement: false,
      layerId: 'source',
      model: 'segment-anything-2-large' as const,
      prompt: 'cat',
    };

    await expect(runSelectObject({ ...options, deps: aborted })).resolves.toEqual({ status: 'aborted' });
    await expect(runSelectObject({ ...options, deps: failed })).resolves.toEqual({
      message: 'SAM graph failed',
      status: 'failed',
    });
  });
});

describe('routeSelectObjectResult', () => {
  const createDeps = () => ({
    commitMaskImageResult: vi.fn<SelectObjectRouteDeps['commitMaskImageResult']>(() =>
      Promise.resolve({ layerId: 'new-mask', status: 'committed' as const })
    ),
    makeImageDurable: vi.fn<SelectObjectRouteDeps['makeImageDurable']>(() => Promise.resolve()),
    replaceSelectionFromImage: vi.fn<SelectObjectRouteDeps['replaceSelectionFromImage']>(() =>
      Promise.resolve({ status: 'selected' as const })
    ),
  });

  it('routes selection without making the intermediate image durable', async () => {
    const deps = createDeps();
    const signal = new AbortController().signal;

    await expect(
      routeSelectObjectResult({ deps, isCurrent: () => true, result: ready, signal, target: 'selection' })
    ).resolves.toEqual({ status: 'selected' });
    expect(deps.replaceSelectionFromImage).toHaveBeenCalledWith(guard, ready.image, rect, signal);
    expect(deps.makeImageDurable).not.toHaveBeenCalled();
    expect(deps.commitMaskImageResult).not.toHaveBeenCalled();
  });

  it('preserves an authoritative selected result when the session invalidates after the engine mutation', async () => {
    const deps = createDeps();
    let current = true;
    deps.replaceSelectionFromImage.mockImplementation(() => {
      current = false;
      return Promise.resolve({ status: 'selected' });
    });

    await expect(
      routeSelectObjectResult({
        deps,
        isCurrent: () => current,
        result: ready,
        signal: new AbortController().signal,
        target: 'selection',
      })
    ).resolves.toEqual({ status: 'selected' });
  });

  it.each(['inpaint_mask', 'regional_guidance'] as const)(
    'makes the image durable before committing %s',
    async (target) => {
      const calls: string[] = [];
      const deps = createDeps();
      deps.makeImageDurable.mockImplementation(() => {
        calls.push('durable');
        return Promise.resolve();
      });
      deps.commitMaskImageResult.mockImplementation((options) => {
        calls.push('commit');
        expect(options).toEqual({ guard, image: ready.image, rect, signal: expect.any(AbortSignal), target });
        return Promise.resolve({ layerId: 'new-mask', status: 'committed' as const });
      });

      await expect(
        routeSelectObjectResult({
          deps,
          isCurrent: () => true,
          result: ready,
          signal: new AbortController().signal,
          target,
        })
      ).resolves.toEqual({ layerId: 'new-mask', status: 'committed' });
      expect(calls).toEqual(['durable', 'commit']);
      expect(deps.replaceSelectionFromImage).not.toHaveBeenCalled();
    }
  );

  it('does not commit when cancellation lands during durability promotion', async () => {
    const durable = createDeferred<void>();
    const controller = new AbortController();
    const deps = createDeps();
    deps.makeImageDurable.mockReturnValue(durable.promise);
    const pending = routeSelectObjectResult({
      deps,
      isCurrent: () => true,
      result: ready,
      signal: controller.signal,
      target: 'inpaint_mask',
    });
    controller.abort();
    durable.resolve();

    await expect(pending).resolves.toEqual({ status: 'aborted' });
    expect(deps.commitMaskImageResult).not.toHaveBeenCalled();
  });

  it('rejects a stale session after durability promotion without committing', async () => {
    const deps = createDeps();
    let current = true;
    deps.makeImageDurable.mockImplementation(() => {
      current = false;
      return Promise.resolve();
    });

    await expect(
      routeSelectObjectResult({
        deps,
        isCurrent: () => current,
        result: ready,
        signal: new AbortController().signal,
        target: 'regional_guidance',
      })
    ).resolves.toEqual({ status: 'stale' });
    expect(deps.commitMaskImageResult).not.toHaveBeenCalled();
  });

  it('preserves an authoritative committed result when the session invalidates after the engine mutation', async () => {
    const deps = createDeps();
    let current = true;
    deps.commitMaskImageResult.mockImplementation(() => {
      current = false;
      return Promise.resolve({ layerId: 'new-mask', status: 'committed' });
    });

    await expect(
      routeSelectObjectResult({
        deps,
        isCurrent: () => current,
        result: ready,
        signal: new AbortController().signal,
        target: 'regional_guidance',
      })
    ).resolves.toEqual({ layerId: 'new-mask', status: 'committed' });
  });

  it('maps durability and dependency failures without a late mutation', async () => {
    const durability = createDeps();
    durability.makeImageDurable.mockRejectedValue(new Error('promotion failed'));
    const selection = createDeps();
    selection.replaceSelectionFromImage.mockRejectedValue(new Error('selection failed'));

    await expect(
      routeSelectObjectResult({
        deps: durability,
        isCurrent: () => true,
        result: ready,
        signal: new AbortController().signal,
        target: 'inpaint_mask',
      })
    ).resolves.toEqual({ message: 'promotion failed', status: 'failed' });
    expect(durability.commitMaskImageResult).not.toHaveBeenCalled();
    await expect(
      routeSelectObjectResult({
        deps: selection,
        isCurrent: () => true,
        result: ready,
        signal: new AbortController().signal,
        target: 'selection',
      })
    ).resolves.toEqual({ message: 'selection failed', status: 'failed' });
  });

  it.each(['aborted', 'missing', 'locked', 'stale', 'unsupported', 'busy'] as const)(
    'forwards engine status %s exhaustively',
    async (status) => {
      const deps = createDeps();
      deps.commitMaskImageResult.mockResolvedValue({ status });

      await expect(
        routeSelectObjectResult({
          deps,
          isCurrent: () => true,
          result: ready,
          signal: new AbortController().signal,
          target: 'inpaint_mask',
        })
      ).resolves.toEqual({ status });
    }
  );
});
