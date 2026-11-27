import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { StubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import type { CanvasUtilityGraphResult } from '@workbench/canvas-operations/contracts';

import { createControlLayer, getDocumentLeaves } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createCanvasEngine } from './createCanvasEngine';
import { getCanvasOperations } from './operationAccess';

const DEBOUNCE_MS = 400;
const SIZE = 64;

const photoControl = (id: string): CanvasControlLayerContract => ({
  ...createControlLayer({ id, name: id }),
  source: { image: { height: SIZE, imageName: `${id}.png`, width: SIZE }, type: 'image' },
});

const shapeRaster = (id: string): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { fill: '#fff', height: SIZE, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: SIZE },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

/** Decodes every image at the test size, so a detected map lands on its input. */
const sizedBackend = (): StubRasterBackend => ({
  ...createTestStubRasterBackend(),
  createImageBitmap: () =>
    Promise.resolve({ close: () => undefined, height: SIZE, width: SIZE } as unknown as ImageBitmap),
});

/** A real engine; graphs stay pending unless `runGraph` answers them. */
const setup = (
  layers: CanvasLayerContract[] = [photoControl('c1')],
  runGraph: (options: { signal?: AbortSignal }) => Promise<CanvasUtilityGraphResult> = () => new Promise(() => {})
) => {
  const store = createWorkbenchStore();
  const projectId = store.getState().activeProjectId;
  store.commands.canvas.apply(projectId, {
    document: { ...createEmptyCanvasDocument(), selectedLayerId: layers[0]!.id, stacks: stacksFrom(layers) },
    type: 'replaceCanvasDocument',
  });
  const graphSignals: AbortSignal[] = [];
  const graph = vi.fn((options: { signal?: AbortSignal }) => {
    graphSignals.push(options.signal!);
    return runGraph(options);
  });
  const makeDurable = vi.fn((_imageName: string) => Promise.resolve());
  const engine = createCanvasEngine({
    backend: sizedBackend(),
    controlDetectionDeps: { makeDurable },
    ensureProjectOnServer: () => Promise.resolve(),
    filterDeps: { runGraph: graph, uploadIntermediate: () => Promise.resolve({ imageName: 'filter-input.png' }) },
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  const operations = getCanvasOperations(engine);
  const leaves = () => getDocumentLeaves(store.queries.getProject(projectId)!.canvas.document);
  const layer = (id = 'c1') => leaves().find((leaf) => leaf.id === id) as CanvasControlLayerContract | undefined;
  return { engine, graph, graphSignals, layer, leaves, makeDurable, operations };
};

afterEach(() => {
  vi.useRealTimers();
});

describe('createCanvasEngine control detection', () => {
  it('detects after the debounce, lands the map and undoes the intent with its map in one step', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { engine, graph, layer, makeDurable, operations } = setup(undefined, () =>
      Promise.resolve({ height: SIZE, imageName: 'edges-map.png', width: SIZE })
    );
    const before = layer();

    expect(operations.applyControlLayerIntent('c1', { controlType: 'edges', kind: 'set-type' })).toBe('updated');
    expect(operations.getControlDetectionStatus('c1')).toEqual({ status: 'queued' });
    await vi.advanceTimersByTimeAsync(DEBOUNCE_MS - 1);
    expect(graph).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    await vi.waitFor(() => expect(operations.getControlDetectionStatus('c1')).toEqual({ status: 'idle' }));

    expect(graph).toHaveBeenCalledOnce();
    expect(makeDurable).toHaveBeenCalledExactlyOnceWith('edges-map.png');
    expect(layer()).toMatchObject({
      controlType: 'edges',
      detection: { type: 'canny_edge_detection' },
      filter: { type: 'canny_edge_detection' },
      original: before!.source,
      source: { bitmap: { imageName: 'edges-map.png' }, type: 'paint' },
    });

    expect(await engine.history.undo()).toBe('applied');
    expect(layer()).toEqual(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(graph).toHaveBeenCalledOnce();
    engine.lifecycle.dispose();
  });

  it('suspends detection on cooldown and resumes it on activation', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { engine, graph, graphSignals, operations } = setup();
    operations.applyControlLayerIntent('c1', { controlType: 'edges', kind: 'set-type' });
    await vi.advanceTimersByTimeAsync(DEBOUNCE_MS);

    void engine.lifecycle.beginCooldown();
    expect(graphSignals[0]!.aborted).toBe(true);
    expect(operations.getControlDetectionStatus('c1')).toEqual({ status: 'queued' });

    engine.lifecycle.activate();
    await vi.advanceTimersByTimeAsync(0);
    expect(graph).toHaveBeenCalledTimes(2);
    engine.lifecycle.dispose();
  });

  it('drops the status of a deleted layer', () => {
    const { engine, operations } = setup();
    operations.applyControlLayerIntent('c1', { controlType: 'edges', kind: 'set-type' });

    const remove = engine.document.model()!.prepare({ ids: ['c1'], type: 'remove' });
    if (remove.status !== 'prepared') {
      throw new Error('expected a prepared removal');
    }
    expect(engine.layers.commitPrepared('Delete', remove.edit).status).toBe('committed');
    expect(operations.getControlDetectionStatus('c1')).toEqual({ status: 'idle' });
    engine.lifecycle.dispose();
  });

  it('aborts running detection on dispose, idempotently', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { engine, graphSignals, operations } = setup();
    const listener = vi.fn();
    operations.subscribeControlDetectionStatus('c1', listener);
    operations.applyControlLayerIntent('c1', { controlType: 'edges', kind: 'set-type' });
    await vi.advanceTimersByTimeAsync(DEBOUNCE_MS);
    expect(graphSignals).toHaveLength(1);
    listener.mockClear();

    engine.lifecycle.dispose();
    engine.lifecycle.dispose();

    expect(graphSignals[0]!.aborted).toBe(true);
    expect(operations.getControlDetectionStatus('c1')).toEqual({ status: 'idle' });
    expect(listener).toHaveBeenCalledOnce();
    expect(operations.applyControlLayerIntent('c1', { kind: 'draw' })).toBe('blocked');
  });

  it('types a canny filter copy saved as control as an edges layer used as-is', async () => {
    const { engine, leaves, operations } = setup([shapeRaster('r1')], () =>
      Promise.resolve({ height: SIZE, imageName: 'canny.png', width: SIZE })
    );
    expect((await engine.exports.exportLayerPixels('r1')).status).toBe('ok');
    expect(operations.startFilterOperation('r1')).toBe('started');
    operations.updateFilterOperation({
      settings: { high_threshold: 200, low_threshold: 100 },
      type: 'canny_edge_detection',
    });
    await expect(operations.processFilterOperation()).resolves.toBe('completed');

    await expect(operations.commitFilterOperation('control', () => Promise.resolve())).resolves.toBe('committed');

    const copies = leaves().filter((leaf) => leaf.type === 'control');
    expect(copies).toHaveLength(1);
    expect(copies[0]).toMatchObject({ controlType: 'edges', detection: null });
    expect(copies[0]).not.toHaveProperty('original');
    engine.lifecycle.dispose();
  });
});
