import type { CanvasLayerContract, CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine, type CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createInitialWorkbenchState, type WorkbenchAction, workbenchReducer } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

const imageLayer = (id: string, imageName: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 10, imageName, width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const maskLayer = (id: string, imageName: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: { height: 10, imageName, width: 10 }, fill: { color: '#e07575', style: 'diagonal' } },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const stagedCandidate: CanvasStagingCandidateContract = {
  height: 10,
  imageName: 'staged-result.png',
  imageUrl: '/staged-result.png',
  placement: { height: 10, opacity: 1, width: 10, x: 0, y: 0 },
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceQueueItemId: 'queue-staged',
  thumbnailUrl: '/staged-result-thumb.png',
  width: 10,
};

const createHarness = (layers: CanvasLayerContract[]) => {
  let state = createInitialWorkbenchState();
  const projectId = state.projects[0]!.id;
  const listeners = new Set<() => void>();
  const dispatch = (action: WorkbenchAction): void => {
    state = workbenchReducer(state, action);
    for (const listener of listeners) {
      listener();
    }
  };
  const apply = (mutation: CanvasProjectMutation): void =>
    dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' });
  apply({
    document: {
      background: 'transparent',
      bbox: { height: 6, width: 6, x: 2, y: 2 },
      height: 100,
      selectedLayerId: layers[0]?.id ?? null,
      stacks: stacksFrom(layers),
      version: 3,
      width: 100,
    },
    type: 'replaceCanvasDocument',
  });
  const getCanvasState = () => state.projects.find((project) => project.id === projectId)?.canvas ?? null;
  const mutationPort: CanvasProjectMutationPort = {
    commitEdit: () => undefined,
    dispatch: (mutation) => {
      const before = getCanvasState();
      apply(mutation);
      return getCanvasState() !== before;
    },
    getCanvasState,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    ensureProjectOnServer: () => Promise.resolve(),
    getMainModelBase: () => null,
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort,
    projectId,
    reportError: () => undefined,
  });
  return { dispatch, engine, projectId };
};

type Harness = ReturnType<typeof createHarness>;

const guardFor = async (engine: CanvasEngine, layerId: string) => {
  const exported = await engine.exports.exportLayerPixels(layerId);
  if (exported.status !== 'ok') {
    throw new Error(`expected exportable ${layerId}, got ${exported.status}`);
  }
  return exported.guard;
};

const warmCaches = async (engine: CanvasEngine, layerIds: readonly string[]): Promise<void> => {
  for (const layerId of layerIds) {
    expect(await engine.previews.requestLayerThumbnail(layerId)).toBe('ready');
  }
};

const resultImage = (imageName: string) => ({ height: 10, imageName, width: 10 });
const resultRect = { height: 10, width: 10, x: 1, y: 1 };

interface HeldCase {
  name: string;
  layers: CanvasLayerContract[];
  act(harness: Harness): unknown;
  /** Names Canvas must hold after the operation (the replaced sources). */
  heldAfterOp: string[];
  /** Names Canvas must hold after undoing it (what redo would restore). */
  heldAfterUndo: string[];
}

const cases: HeldCase[] = [
  {
    act: async ({ engine }) => {
      await warmCaches(engine, ['a']);
      return engine.layers.cropLayerToBbox('a');
    },
    heldAfterOp: ['original-crop.png'],
    heldAfterUndo: ['original-crop.png'],
    layers: [imageLayer('a', 'original-crop.png')],
    name: 'crop',
  },
  {
    act: async ({ engine }) =>
      engine.layers.commitRasterFilterResult({
        guard: await guardFor(engine, 'a'),
        image: resultImage('filter-replace-result.png'),
        mode: 'replace',
        rect: resultRect,
      }),
    heldAfterOp: ['original-filter-replace.png'],
    heldAfterUndo: ['filter-replace-result.png'],
    layers: [imageLayer('a', 'original-filter-replace.png')],
    name: 'filter result replace',
  },
  {
    act: async ({ engine }) =>
      engine.layers.commitRasterFilterResult({
        guard: await guardFor(engine, 'a'),
        image: resultImage('filter-copy-result.png'),
        mode: 'copy',
        rect: resultRect,
      }),
    heldAfterOp: ['filter-copy-result.png'],
    heldAfterUndo: ['filter-copy-result.png'],
    layers: [imageLayer('a', 'original-filter-copy.png')],
    name: 'filter result copy',
  },
  {
    act: async ({ engine }) =>
      engine.layers.commitGeneratedImageResult({
        guard: await guardFor(engine, 'a'),
        image: resultImage('generated-replace-result.png'),
        origin: { x: 1, y: 1 },
        target: 'replace',
      }),
    heldAfterOp: ['original-generated-replace.png'],
    heldAfterUndo: ['generated-replace-result.png'],
    layers: [imageLayer('a', 'original-generated-replace.png')],
    name: 'generated result replace',
  },
  {
    act: async ({ engine }) =>
      engine.layers.commitGeneratedImageResult({
        guard: await guardFor(engine, 'a'),
        image: resultImage('generated-copy-result.png'),
        origin: { x: 1, y: 1 },
        target: 'copy-raster',
      }),
    heldAfterOp: ['generated-copy-result.png'],
    heldAfterUndo: ['generated-copy-result.png'],
    layers: [imageLayer('a', 'original-generated-copy.png')],
    name: 'generated result copy',
  },
  {
    act: async ({ engine }) => {
      await warmCaches(engine, ['upper', 'below']);
      return engine.layers.mergeSelectedRasterLayers(['upper', 'below']);
    },
    heldAfterOp: ['original-merge-selected-upper.png', 'original-merge-selected-below.png'],
    heldAfterUndo: ['original-merge-selected-upper.png', 'original-merge-selected-below.png'],
    layers: [
      imageLayer('upper', 'original-merge-selected-upper.png'),
      imageLayer('below', 'original-merge-selected-below.png'),
    ],
    name: 'merge selected',
  },
  {
    act: async ({ engine }) => {
      await warmCaches(engine, ['upper', 'below']);
      return engine.layers.mergeVisibleRasterLayers();
    },
    heldAfterOp: ['original-merge-visible-upper.png', 'original-merge-visible-below.png'],
    heldAfterUndo: ['original-merge-visible-upper.png', 'original-merge-visible-below.png'],
    layers: [
      imageLayer('upper', 'original-merge-visible-upper.png'),
      imageLayer('below', 'original-merge-visible-below.png'),
    ],
    name: 'merge visible',
  },
  {
    act: async ({ engine }) => {
      await warmCaches(engine, ['a']);
      const live = getDocumentLeaves(engine.document.getDocument()!)[0]!;
      if (live.type !== 'raster') {
        throw new Error('expected a raster conversion source');
      }
      return engine.layers.commitLayerConversion('Convert layer', live, {
        ...live,
        adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
        type: 'control',
        withTransparencyEffect: false,
      });
    },
    heldAfterOp: ['original-convert.png'],
    heldAfterUndo: ['original-convert.png'],
    layers: [imageLayer('a', 'original-convert.png')],
    name: 'convert layer type',
  },
  {
    act: ({ dispatch, engine, projectId }) => {
      dispatch({ candidate: stagedCandidate, projectId, type: 'appendCanvasStagingCandidate' });
      return engine.layers.commitStagedImage({ candidate: stagedCandidate, selectedImageIndex: 0 });
    },
    heldAfterOp: ['staged-result.png'],
    heldAfterUndo: ['staged-result.png'],
    layers: [imageLayer('a', 'original-staged.png')],
    name: 'staged result accept',
  },
  {
    act: async ({ engine }) => {
      await warmCaches(engine, ['mask']);
      return engine.layers.clearMask('mask');
    },
    heldAfterOp: ['original-mask-clear.png'],
    heldAfterUndo: ['original-mask-clear.png'],
    layers: [maskLayer('mask', 'original-mask-clear.png')],
    name: 'mask clear',
  },
  {
    act: async ({ engine }) =>
      engine.layers.commitMaskImageResult({
        guard: await guardFor(engine, 'a'),
        image: resultImage('mask-result.png'),
        rect: resultRect,
        target: 'inpaint_mask',
      }),
    heldAfterOp: ['mask-result.png'],
    heldAfterUndo: ['mask-result.png'],
    layers: [imageLayer('a', 'original-mask-result.png')],
    name: 'mask result',
  },
  {
    act: async ({ engine }) => {
      await warmCaches(engine, ['a']);
      const source = getDocumentLeaves(engine.document.getDocument()!)[0]!;
      return engine.layers.commitLayerCopy(
        'Copy layer',
        'a',
        { ...structuredClone(source), id: 'a-copy', name: 'a copy' },
        engine.document.captureInsertionAnchor('raster', 'a')
      );
    },
    heldAfterOp: ['original-copy.png'],
    heldAfterUndo: ['original-copy.png'],
    layers: [imageLayer('a', 'original-copy.png')],
    name: 'copy layer',
  },
  {
    act: async ({ engine }) => {
      await warmCaches(engine, ['a']);
      return engine.layers.duplicateLayers(['a']);
    },
    heldAfterOp: ['original-duplicate.png'],
    heldAfterUndo: ['original-duplicate.png'],
    layers: [imageLayer('a', 'original-duplicate.png')],
    name: 'duplicate layer',
  },
];

describe('Canvas history holds resurrectable media', () => {
  it.each(cases)('$name', async ({ act, heldAfterOp, heldAfterUndo, layers }) => {
    const harness = createHarness(layers);
    const { engine } = harness;

    await act(harness);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.history.getHeldAssetRefs().images).toEqual(expect.arrayContaining(heldAfterOp));

    await engine.history.undo();
    expect(engine.stores.canRedo.get()).toBe(true);
    expect(engine.history.getHeldAssetRefs().images).toEqual(expect.arrayContaining(heldAfterUndo));

    engine.history.clearHistory();
    expect(engine.history.getHeldAssetRefs().images).toEqual([]);
    engine.lifecycle.dispose();
  });
});
