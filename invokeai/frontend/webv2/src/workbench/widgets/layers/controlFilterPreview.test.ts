import type { ExecuteCompositePlanDeps } from '@workbench/canvas-engine/export/compositeForGeneration';
import type { CanvasControlLayerContract, CanvasDocumentContractV2 } from '@workbench/types';

import { describe, expect, it, vi } from 'vitest';

import type { RunControlFilterPreviewDeps } from './controlFilterPreview';

// The pixel-level composite is exercised in compositeForGeneration.test; here we
// stub it so the orchestration (plan → composite → filter graph → utility queue)
// is tested in isolation against a fixed uploaded composite.
vi.mock('@workbench/canvas-engine/export/compositeForGeneration', () => ({
  executeControlComposite: vi.fn(() =>
    Promise.resolve({
      height: 100,
      imageName: 'composite.png',
      key: 'k',
      pixelHash: 'p',
      reusedUpload: false,
      width: 100,
    })
  ),
}));

import { executeControlComposite } from '@workbench/canvas-engine/export/compositeForGeneration';

import { ControlFilterPreviewError, runControlFilterPreview } from './controlFilterPreview';

const executeControlCompositeMock = vi.mocked(executeControlComposite);

const controlLayer = (id: string, hasContent = true): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: hasContent
    ? { image: { height: 48, imageName: `${id}-src`, width: 64 }, type: 'image' }
    : { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

/** A paint-source control layer whose content bitmap sits at a non-zero offset. */
const offsetPaintControlLayer = (id: string): CanvasControlLayerContract => ({
  ...controlLayer(id),
  source: { bitmap: { height: 30, imageName: `${id}-paint`, width: 20 }, offset: { x: 15, y: 25 }, type: 'paint' },
});

const makeDoc = (layers: CanvasControlLayerContract[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 100,
});

/** Executor deps are unused here (the composite is mocked) — a trivial stub suffices. */
const fakeExecutorDeps = (): ExecuteCompositePlanDeps =>
  ({ dedupe: { byHash: new Map(), byKey: new Map() } }) as unknown as ExecuteCompositePlanDeps;

const baseDeps = (overrides: Partial<RunControlFilterPreviewDeps> = {}): RunControlFilterPreviewDeps => ({
  executorDeps: fakeExecutorDeps(),
  flushPendingUploads: () => Promise.resolve(),
  getDocument: () => makeDoc([controlLayer('c1')]),
  runFilterGraph: () => Promise.resolve('filtered.png'),
  ...overrides,
});

describe('runControlFilterPreview', () => {
  it('flushes, composites, builds the filter graph, and returns the filtered image + composite dims', async () => {
    const runFilterGraph = vi.fn((graph, outputNodeId: string) => {
      // The graph is the single-node filter over the uploaded composite.
      expect(outputNodeId).toBe('control_filter');
      expect(graph.nodes.control_filter?.type).toBe('canny_edge_detection');
      expect(graph.nodes.control_filter?.image).toEqual({ image_name: 'composite.png' });
      return Promise.resolve('filtered.png');
    });
    const flushPendingUploads = vi.fn(() => Promise.resolve());

    const result = await runControlFilterPreview({
      deps: baseDeps({ flushPendingUploads, runFilterGraph }),
      filterType: 'canny_edge_detection',
      layerId: 'c1',
    });

    expect(flushPendingUploads).toHaveBeenCalledOnce();
    expect(runFilterGraph).toHaveBeenCalledOnce();
    expect(result).toEqual({ height: 100, imageName: 'filtered.png', origin: { x: 0, y: 0 }, width: 100 });
  });

  it('scopes the composite to the LAYER content rect (not the doc bbox) with an identity transform', async () => {
    executeControlCompositeMock.mockClear();
    // c1 is a 64×48 image source at the layer origin, inside a 100×100 doc/bbox.
    await runControlFilterPreview({
      deps: baseDeps(),
      filterType: 'canny_edge_detection',
      layerId: 'c1',
    });

    const entry = executeControlCompositeMock.mock.calls.at(-1)?.[0];
    // The composite bbox is the layer's content rect — 64×48 at origin — NOT the
    // 100×100 doc bbox, so the preview isn't stretched/offset against the overlay.
    expect(entry?.bbox).toEqual({ height: 48, width: 64, x: 0, y: 0 });
    // The layer transform is applied by the compositor/Apply, so the composite
    // itself must use an identity transform (baking it here would double it).
    expect(entry?.layers[0]?.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
  });

  it('returns the content-rect origin for an offset paint control layer (Apply repositions to it)', async () => {
    const result = await runControlFilterPreview({
      deps: baseDeps({ getDocument: () => makeDoc([offsetPaintControlLayer('c1')]) }),
      filterType: 'canny_edge_detection',
      layerId: 'c1',
    });

    // The bitmap sits at { x: 15, y: 25 }; the composite is scoped to that rect and
    // the origin travels back so Apply positions the filtered image there.
    const entry = executeControlCompositeMock.mock.calls.at(-1)?.[0];
    expect(entry?.bbox).toEqual({ height: 30, width: 20, x: 15, y: 25 });
    expect(result.origin).toEqual({ x: 15, y: 25 });
  });

  it('passes custom filter settings into the graph', async () => {
    const runFilterGraph = vi.fn((graph) => {
      expect(graph.nodes.control_filter?.low_threshold).toBe(40);
      return Promise.resolve('filtered.png');
    });
    await runControlFilterPreview({
      deps: baseDeps({ runFilterGraph }),
      filterType: 'canny_edge_detection',
      layerId: 'c1',
      settings: { high_threshold: 200, low_threshold: 40 },
    });
    expect(runFilterGraph).toHaveBeenCalledOnce();
  });

  it('rejects with a ControlFilterPreviewError when the layer has no content', async () => {
    await expect(
      runControlFilterPreview({
        deps: baseDeps({ getDocument: () => makeDoc([controlLayer('c1', false)]) }),
        filterType: 'canny_edge_detection',
        layerId: 'c1',
      })
    ).rejects.toBeInstanceOf(ControlFilterPreviewError);
  });

  it('rejects with a ControlFilterPreviewError when the layer is absent', async () => {
    await expect(
      runControlFilterPreview({
        deps: baseDeps(),
        filterType: 'canny_edge_detection',
        layerId: 'missing',
      })
    ).rejects.toBeInstanceOf(ControlFilterPreviewError);
  });

  it('rejects with AbortError when the signal aborts before the filter runs', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      runControlFilterPreview({
        deps: baseDeps(),
        filterType: 'canny_edge_detection',
        layerId: 'c1',
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('does not run the filter graph once aborted', async () => {
    const controller = new AbortController();
    const runFilterGraph = vi.fn(() => Promise.resolve('filtered.png'));
    await expect(
      runControlFilterPreview({
        deps: baseDeps({
          flushPendingUploads: () => {
            controller.abort();
            return Promise.resolve();
          },
          runFilterGraph,
        }),
        filterType: 'canny_edge_detection',
        layerId: 'c1',
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(runFilterGraph).not.toHaveBeenCalled();
  });
});
