import type { CommitGeneratedImageResult, LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { GalleryImage } from '@workbench/gallery/api';
import type { WorkflowImageBinding } from '@workbench/workflows/layerWorkflow';
import type { InvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import type { ProjectGraphState } from '@workbench/workflows/types';

import { describe, expect, it, vi } from 'vitest';

import {
  runLayerWorkflow,
  type LayerWorkflowDestination,
  type RunLayerWorkflowDeps,
  type RunLayerWorkflowOptions,
} from './runLayerWorkflow';

const guard = {
  cacheVersion: 1,
  documentGeneration: 2,
  layer: { id: 'layer-1' },
  layerId: 'layer-1',
  projectId: 'project-1',
} as LayerExportGuard;
const document = { id: 'workflow-1' } as ProjectGraphState;
const templatesSnapshot = { error: null, status: 'loaded', templates: {} } as InvocationTemplatesSnapshot;
const input: WorkflowImageBinding = { fieldName: 'image', label: 'Input · Image', nodeId: 'input-node' };
const output: WorkflowImageBinding = { fieldName: 'image', label: 'Output · Image', nodeId: 'output-node' };
const sourceBlob = new Blob(['source'], { type: 'image/png' });
const image: GalleryImage = {
  boardId: 'none',
  height: 160,
  imageCategory: 'other',
  imageName: 'workflow-output.png',
  imageUrl: '/full/workflow-output.png',
  queuedAt: '2026-07-09T12:00:00.000Z',
  sourceQueueItemId: 'backend-gallery',
  starred: false,
  thumbnailUrl: '/thumb/workflow-output.png',
  width: 240,
};

const createDeps = () => {
  const order: string[] = [];
  const deps: RunLayerWorkflowDeps = {
    appendStaging: vi.fn(() => order.push('append-staging')),
    buildGraph: vi.fn(() => {
      order.push('build-graph');
      return { graph: { edges: [], id: 'layer-workflow', nodes: {} }, outputNodeId: 'capture' };
    }),
    commitGenerated: vi.fn(() => {
      order.push('commit-generated');
      return Promise.resolve<CommitGeneratedImageResult>({ layerId: 'result-layer', status: 'committed' });
    }),
    createRequestId: vi.fn(() => 'request-7'),
    exportLayer: vi.fn<RunLayerWorkflowDeps['exportLayer']>(() => {
      order.push('export');
      return Promise.resolve({ blob: sourceBlob, guard, rect: { height: 60, width: 80, x: 11, y: 22 }, status: 'ok' });
    }),
    getImage: vi.fn(() => {
      order.push('get-image');
      return Promise.resolve(image);
    }),
    isGuardCurrent: vi.fn(() => true),
    makeDurable: vi.fn(() => {
      order.push('make-durable');
      return Promise.resolve();
    }),
    runGraph: vi.fn<RunLayerWorkflowDeps['runGraph']>(() => {
      order.push('run-graph');
      return Promise.resolve({ imageName: image.imageName, origin: 'webv2:util:test' });
    }),
    saveToGallery: vi.fn<RunLayerWorkflowDeps['saveToGallery']>(() => {
      order.push('save-gallery');
      return Promise.resolve({ ...image, imageCategory: 'general' as const });
    }),
    touchGallery: vi.fn(() => order.push('touch-gallery')),
    uploadIntermediate: vi.fn(() => {
      order.push('upload');
      return Promise.resolve({ imageName: 'layer-input.png' });
    }),
  };
  return { deps, order };
};

const createOptions = (
  destination: LayerWorkflowDestination,
  deps: RunLayerWorkflowDeps,
  signal?: AbortSignal
): RunLayerWorkflowOptions => ({
  deps,
  destination,
  document,
  input,
  layerId: 'layer-1',
  output,
  projectId: 'project-1',
  signal,
  templatesSnapshot,
});

describe('runLayerWorkflow', () => {
  it('runs the captured output to Gallery and refreshes the captured project', async () => {
    const { deps, order } = createDeps();

    await expect(runLayerWorkflow(createOptions('gallery', deps))).resolves.toEqual({
      imageName: image.imageName,
      status: 'completed',
    });

    expect(order).toEqual([
      'export',
      'upload',
      'build-graph',
      'run-graph',
      'get-image',
      'save-gallery',
      'touch-gallery',
    ]);
    expect(deps.buildGraph).toHaveBeenCalledWith({
      document,
      imageName: 'layer-input.png',
      input,
      output,
      templatesSnapshot,
    });
    expect(deps.touchGallery).toHaveBeenCalledWith('project-1');
    expect(deps.makeDurable).not.toHaveBeenCalled();
  });

  it('appends a durable orphan staging candidate at the source origin and native result dimensions', async () => {
    const { deps, order } = createDeps();

    await expect(runLayerWorkflow(createOptions('staging', deps))).resolves.toEqual({
      imageName: image.imageName,
      status: 'completed',
    });

    expect(order).toEqual([
      'export',
      'upload',
      'build-graph',
      'run-graph',
      'get-image',
      'make-durable',
      'append-staging',
    ]);
    expect(deps.appendStaging).toHaveBeenCalledWith('project-1', {
      ...image,
      placement: { height: 160, opacity: 1, width: 240, x: 11, y: 22 },
      sourceQueueItemId: 'layer-workflow:request-7',
    });
  });

  it.each([
    ['replace', 'replace'],
    ['copy-raster', 'copy-raster'],
  ] as const)('promotes then commits the %s destination', async (destination, target) => {
    const { deps, order } = createDeps();

    await expect(runLayerWorkflow(createOptions(destination, deps))).resolves.toEqual({
      imageName: image.imageName,
      status: 'completed',
    });

    expect(order).toEqual([
      'export',
      'upload',
      'build-graph',
      'run-graph',
      'get-image',
      'make-durable',
      'commit-generated',
    ]);
    expect(deps.commitGenerated).toHaveBeenCalledWith({
      guard,
      image: { height: image.height, imageName: image.imageName, width: image.width },
      origin: { x: 11, y: 22 },
      signal: undefined,
      target,
    });
  });

  it.each(['missing', 'disabled', 'unsupported', 'empty', 'not-ready'] as const)(
    'maps an export %s result without starting the workflow',
    async (status) => {
      const { deps } = createDeps();
      vi.mocked(deps.exportLayer).mockResolvedValue({ status });

      await expect(runLayerWorkflow(createOptions('gallery', deps))).resolves.toEqual({ status });

      expect(deps.uploadIntermediate).not.toHaveBeenCalled();
    }
  );

  it('stops with stale after hydration when the source guard changed', async () => {
    const { deps } = createDeps();
    vi.mocked(deps.isGuardCurrent).mockReturnValueOnce(true).mockReturnValueOnce(true).mockReturnValueOnce(false);

    await expect(runLayerWorkflow(createOptions('staging', deps))).resolves.toEqual({ status: 'stale' });

    expect(deps.makeDurable).not.toHaveBeenCalled();
    expect(deps.appendStaging).not.toHaveBeenCalled();
  });

  it('does not append staging when the guard changes during durability', async () => {
    const { deps } = createDeps();
    vi.mocked(deps.isGuardCurrent).mockImplementation(() => vi.mocked(deps.makeDurable).mock.calls.length === 0);

    await expect(runLayerWorkflow(createOptions('staging', deps))).resolves.toEqual({ status: 'stale' });

    expect(deps.makeDurable).toHaveBeenCalledOnce();
    expect(deps.appendStaging).not.toHaveBeenCalled();
  });

  it('checks cancellation at every async boundary and forwards the signal', async () => {
    const controller = new AbortController();
    const { deps } = createDeps();
    vi.mocked(deps.getImage).mockImplementation((_imageName, signal) => {
      expect(signal).toBe(controller.signal);
      controller.abort();
      return Promise.resolve(image);
    });

    await expect(runLayerWorkflow(createOptions('staging', deps, controller.signal))).resolves.toEqual({
      status: 'aborted',
    });

    expect(deps.uploadIntermediate).toHaveBeenCalledWith(sourceBlob, controller.signal);
    expect(deps.runGraph).toHaveBeenCalledWith(
      expect.objectContaining({ outputNodeId: 'capture', signal: controller.signal })
    );
    expect(deps.makeDurable).not.toHaveBeenCalled();
  });

  it.each(['export', 'upload', 'run', 'hydrate', 'durability'] as const)(
    'stops routing when canceled during %s',
    async (boundary) => {
      const controller = new AbortController();
      const { deps } = createDeps();

      if (boundary === 'export') {
        vi.mocked(deps.exportLayer).mockImplementation(() => {
          controller.abort();
          return Promise.resolve({
            blob: sourceBlob,
            guard,
            rect: { height: 60, width: 80, x: 11, y: 22 },
            status: 'ok',
          });
        });
      } else if (boundary === 'upload') {
        vi.mocked(deps.uploadIntermediate).mockImplementation(() => {
          controller.abort();
          return Promise.resolve({ imageName: 'layer-input.png' });
        });
      } else if (boundary === 'run') {
        vi.mocked(deps.runGraph).mockImplementation(() => {
          controller.abort();
          return Promise.resolve({ imageName: image.imageName, origin: 'webv2:util:test' });
        });
      } else if (boundary === 'hydrate') {
        vi.mocked(deps.getImage).mockImplementation(() => {
          controller.abort();
          return Promise.resolve(image);
        });
      } else {
        vi.mocked(deps.makeDurable).mockImplementation(() => {
          controller.abort();
          return Promise.resolve();
        });
      }

      await expect(runLayerWorkflow(createOptions('staging', deps, controller.signal))).resolves.toEqual({
        status: 'aborted',
      });
      expect(deps.appendStaging).not.toHaveBeenCalled();
      expect(deps.commitGenerated).not.toHaveBeenCalled();
    }
  );

  it('does not route when durability promotion fails', async () => {
    const { deps } = createDeps();
    vi.mocked(deps.makeDurable).mockRejectedValue(new Error('promotion failed'));

    await expect(runLayerWorkflow(createOptions('staging', deps))).resolves.toEqual({
      message: 'promotion failed',
      stage: 'durability',
      status: 'failed',
    });
    expect(deps.appendStaging).not.toHaveBeenCalled();
  });

  it('does not refresh Gallery when saving the output fails', async () => {
    const { deps } = createDeps();
    vi.mocked(deps.saveToGallery).mockRejectedValue(new Error('save failed'));

    await expect(runLayerWorkflow(createOptions('gallery', deps))).resolves.toEqual({
      message: 'save failed',
      stage: 'gallery',
      status: 'failed',
    });
    expect(deps.touchGallery).not.toHaveBeenCalled();
  });

  it('preserves an authoritative successful engine commit after a late abort', async () => {
    const controller = new AbortController();
    const { deps } = createDeps();
    vi.mocked(deps.commitGenerated).mockImplementation(() => {
      controller.abort();
      return Promise.resolve({ layerId: 'result-layer', status: 'committed' });
    });

    await expect(runLayerWorkflow(createOptions('replace', deps, controller.signal))).resolves.toEqual({
      imageName: image.imageName,
      status: 'completed',
    });
  });

  it.each(['missing', 'locked', 'stale', 'unsupported', 'busy', 'aborted'] as const)(
    'maps a generated-image commit %s result',
    async (status) => {
      const { deps } = createDeps();
      vi.mocked(deps.commitGenerated).mockResolvedValue({ status });

      await expect(runLayerWorkflow(createOptions('replace', deps))).resolves.toEqual({ status });
    }
  );

  it('maps generated-image commit failures and unexpected API failures', async () => {
    const commit = createDeps();
    vi.mocked(commit.deps.commitGenerated).mockResolvedValue({ message: 'draw failed', status: 'failed' });
    await expect(runLayerWorkflow(createOptions('replace', commit.deps))).resolves.toEqual({
      message: 'draw failed',
      stage: 'commit',
      status: 'failed',
    });

    const hydrate = createDeps();
    vi.mocked(hydrate.deps.getImage).mockRejectedValue(new Error('metadata failed'));
    await expect(runLayerWorkflow(createOptions('gallery', hydrate.deps))).resolves.toEqual({
      message: 'metadata failed',
      stage: 'hydrate',
      status: 'failed',
    });
  });

  it('identifies graph-build failures for specific UI feedback', async () => {
    const { deps } = createDeps();
    vi.mocked(deps.buildGraph).mockImplementation(() => {
      throw new Error('workflow is not ready');
    });

    await expect(runLayerWorkflow(createOptions('gallery', deps))).resolves.toEqual({
      message: 'workflow is not ready',
      stage: 'graph',
      status: 'failed',
    });
  });

  it('maps AbortError failures to aborted', async () => {
    const { deps } = createDeps();
    vi.mocked(deps.runGraph).mockRejectedValue(new DOMException('canceled', 'AbortError'));

    await expect(runLayerWorkflow(createOptions('gallery', deps))).resolves.toEqual({ status: 'aborted' });
  });
});
