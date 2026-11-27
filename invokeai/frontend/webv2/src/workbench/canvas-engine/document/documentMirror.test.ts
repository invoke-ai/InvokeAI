import type {
  CanvasDocumentContractV3,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStagingAreaContractV2,
  CanvasStateContractV3,
  CanvasNodeContract,
} from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';

import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import type { DocumentMirrorCallbacks } from './documentMirror';

import { getDocumentLeaves } from './documentIndex';
import { createDocumentMirror, getForestDiffCount, resetForestDiffCount } from './documentMirror';

const rasterLayer = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const makeDoc = (
  layers: CanvasNodeContract[],
  overrides: Partial<CanvasDocumentContractV3> = {}
): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
  width: 100,
  ...overrides,
});

const makeStaging = (): CanvasStagingAreaContractV2 => ({
  areThumbnailsVisible: false,
  autoSwitchMode: 'off',
  isVisible: false,
  pendingImageIds: [],
  pendingImages: [],
  selectedImageIndex: 0,
});

const makeCanvas = (document: CanvasDocumentContractV3, documentRevision = 0): CanvasStateContractV3 => ({
  document,
  documentRevision,
  snapshots: [],
  stagingArea: makeStaging(),
  version: 3,
});

interface FakeProject {
  id: string;
  canvas: CanvasStateContractV3;
}

const createFakeStore = (projects: FakeProject[]) => {
  let state = { projects };
  const listeners = new Set<() => void>();
  return {
    getState: () => state,
    replaceStateSilently: (next: { projects: FakeProject[] }) => {
      state = next;
    },
    setState: (next: { projects: FakeProject[] }) => {
      state = next;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
};

const spyCallbacks = () => ({
  onBboxChanged: vi.fn<DocumentMirrorCallbacks['onBboxChanged']>(),
  onDocumentReplaced: vi.fn<DocumentMirrorCallbacks['onDocumentReplaced']>(),
  onLayerOrderChanged: vi.fn<DocumentMirrorCallbacks['onLayerOrderChanged']>(),
  onLayersChanged: vi.fn<DocumentMirrorCallbacks['onLayersChanged']>(),
  onLayersRecomposite: vi.fn<NonNullable<DocumentMirrorCallbacks['onLayersRecomposite']>>(),
  onSelectionChanged: vi.fn<NonNullable<DocumentMirrorCallbacks['onSelectionChanged']>>(),
});

describe('createDocumentMirror: selection changes', () => {
  const withSelection = (doc: CanvasDocumentContractV3, selectedLayerId: string | null): CanvasDocumentContractV3 => ({
    ...doc,
    selectedLayerId,
  });

  it('reports a selection-only change without reporting a layer change', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const doc = makeDoc([a, b], { selectedLayerId: 'a' });
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const nextDoc = withSelection(doc, 'b');
    expect(nextDoc.stacks).toBe(doc.stacks);
    store.setState({ projects: [{ canvas: { ...canvas, document: nextDoc }, id: 'p1' }] });

    expect(callbacks.onSelectionChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onSelectionChanged).toHaveBeenCalledWith('b');
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onLayerOrderChanged).not.toHaveBeenCalled();
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
    expect(callbacks.onBboxChanged).not.toHaveBeenCalled();
  });

  it('reports the cleared selection when the selected layer is removed', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const doc = makeDoc([a, b], { selectedLayerId: 'a' });
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const nextDoc: CanvasDocumentContractV3 = { ...doc, stacks: stacksFrom([b]), selectedLayerId: 'b' };
    store.setState({ projects: [{ canvas: { ...canvas, document: nextDoc }, id: 'p1' }] });

    expect(callbacks.onLayersChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onSelectionChanged).toHaveBeenCalledWith('b');
  });
});

describe('createDocumentMirror', () => {
  it('synchronously refreshes from authoritative state when normal notification was interrupted', () => {
    const a = rasterLayer('a');
    const doc = makeDoc([a]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    const mirror = createDocumentMirror(store, 'p1', callbacks);
    const updated: CanvasDocumentContractV3 = { ...doc, stacks: stacksFrom([{ ...a, opacity: 0.5 }]) };

    store.replaceStateSilently({ projects: [{ canvas: { ...canvas, document: updated }, id: 'p1' }] });
    expect(mirror.getDocument()).toBe(doc);

    mirror.refresh();

    expect(mirror.getDocument()).toBe(updated);
    expect(callbacks.onLayersChanged).toHaveBeenCalledWith(['a'], [], false);
  });

  it('reports exactly the edited layer id when one layer changes by reference', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const doc = makeDoc([a, b]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const nextDoc: CanvasDocumentContractV3 = { ...doc, stacks: stacksFrom([{ ...a, opacity: 0.5 }, b]) };
    store.setState({ projects: [{ canvas: { ...canvas, document: nextDoc }, id: 'p1' }] });

    expect(callbacks.onLayersChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).toHaveBeenCalledWith(['a'], [], false);
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
    expect(callbacks.onBboxChanged).not.toHaveBeenCalled();
  });

  it('reports a prop-only edit as changed but not source-changed, and a source swap as both', () => {
    const a = rasterLayer('a');
    const doc = makeDoc([a]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const opacityEdit: CanvasDocumentContractV3 = { ...doc, stacks: stacksFrom([{ ...a, opacity: 0.5 }]) };
    store.setState({ projects: [{ canvas: { ...canvas, document: opacityEdit }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a'], [], false);

    const swapped = getDocumentLeaves(store.getState().projects[0]!.canvas.document)[0] as CanvasRasterLayerContractV2;
    const sourceSwap: CanvasDocumentContractV3 = {
      ...doc,
      stacks: stacksFrom([
        { ...swapped, source: { image: { height: 10, imageName: 'a-v2', width: 10 }, type: 'image' as const } },
      ]),
    };
    store.setState({ projects: [{ canvas: { ...canvas, document: sourceSwap }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a'], ['a'], false);
  });

  it('for a mask: a fill-only change is NOT source-changed, a bitmap swap IS (protects unflushed strokes)', () => {
    const mask: CanvasInpaintMaskLayerContract = {
      blendMode: 'normal',
      id: 'm',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
      name: 'm',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    };
    const doc = makeDoc([mask]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const fillEdit: CanvasDocumentContractV3 = {
      ...doc,
      stacks: stacksFrom([{ ...mask, mask: { bitmap: null, fill: { color: '#00ff00', style: 'grid' as const } } }]),
    };
    store.setState({ projects: [{ canvas: { ...canvas, document: fillEdit }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['m'], [], false);

    const current = getDocumentLeaves(
      store.getState().projects[0]!.canvas.document
    )[0] as CanvasInpaintMaskLayerContract;
    const bitmapSwap: CanvasDocumentContractV3 = {
      ...doc,
      stacks: stacksFrom([
        { ...current, mask: { ...current.mask, bitmap: { height: 20, imageName: 'mask-v1', width: 30 } } },
      ]),
    };
    store.setState({ projects: [{ canvas: { ...canvas, document: bitmapSwap }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['m'], ['m'], false);
  });

  it('reports added and removed layer ids', () => {
    const a = rasterLayer('a');
    const doc = makeDoc([a]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const b = rasterLayer('b');
    store.setState({
      projects: [{ canvas: { ...canvas, document: { ...doc, stacks: stacksFrom([b, a]) } }, id: 'p1' }],
    });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['b'], ['b'], true);

    store.setState({ projects: [{ canvas: { ...canvas, document: { ...doc, stacks: stacksFrom([b]) } }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a'], [], true);
  });

  it('fires onLayerOrderChanged exactly once on a pure reorder, with no layer ids reported', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const doc = makeDoc([a, b]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    store.setState({
      projects: [{ canvas: { ...canvas, document: { ...doc, stacks: stacksFrom([b, a]) } }, id: 'p1' }],
    });

    expect(callbacks.onLayerOrderChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
    expect(callbacks.onBboxChanged).not.toHaveBeenCalled();
  });

  it('fires onDocumentReplaced when dimensions change', () => {
    const doc = makeDoc([rasterLayer('a')]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    store.setState({
      projects: [{ canvas: { ...canvas, document: { ...doc, width: 200 } }, id: 'p1' }],
    });
    expect(callbacks.onDocumentReplaced).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
  });

  it('fires onDocumentReplaced when documentRevision changes, even with identical dims and layer ids', () => {
    const doc = makeDoc([rasterLayer('a')]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    // A same-dims snapshot restore: structuredClone reuses layer ids and keeps width/height/background, so only the
    // revision bump signals the swap.
    const restored = structuredClone(doc);
    store.setState({ projects: [{ canvas: makeCanvas(restored, 1), id: 'p1' }] });

    expect(callbacks.onDocumentReplaced).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onLayerOrderChanged).not.toHaveBeenCalled();
  });

  it('is silent when an unrelated project changes (identity short-circuit)', () => {
    const docA = makeDoc([rasterLayer('a')]);
    const canvasA = makeCanvas(docA);
    const docB = makeDoc([rasterLayer('z')]);
    const canvasB = makeCanvas(docB);
    const store = createFakeStore([
      { canvas: canvasA, id: 'p1' },
      { canvas: canvasB, id: 'p2' },
    ]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    store.setState({
      projects: [
        { canvas: canvasA, id: 'p1' },
        { canvas: makeCanvas(makeDoc([rasterLayer('z', { opacity: 0.2 })])), id: 'p2' },
      ],
    });
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
    expect(callbacks.onBboxChanged).not.toHaveBeenCalled();
  });

  it('stops observing after dispose', () => {
    const doc = makeDoc([rasterLayer('a')]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    const mirror = createDocumentMirror(store, 'p1', callbacks);

    mirror.dispose();
    store.setState({ projects: [{ canvas: makeCanvas(makeDoc([rasterLayer('a', { opacity: 0.1 })])), id: 'p1' }] });
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
  });
});

describe('createDocumentMirror: groups', () => {
  const setup = (doc: CanvasDocumentContractV3) => {
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);
    const set = (next: CanvasDocumentContractV3) =>
      store.setState({ projects: [{ canvas: { ...canvas, document: next }, id: 'p1' }] });
    return { callbacks, set };
  };

  it('reports every descendant leaf as changed but not source-changed when a group flag flips', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const c = rasterLayer('c');
    const doc = makeDoc([groupContract('g', [a, groupContract('h', [b])]), c]);
    const { callbacks, set } = setup(doc);

    set({ ...doc, stacks: stacksFrom([groupContract('g', [a, groupContract('h', [b])], { isEnabled: false }), c]) });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a', 'b'], [], false);
    expect(callbacks.onLayerOrderChanged).not.toHaveBeenCalled();
  });

  it('fans a group adjustment-stack edit out to descendants on the non-destructive recomposite channel', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const c = rasterLayer('c');
    const stack = [{ brightness: 0.3, contrast: 0, id: 'ga', isEnabled: true, type: 'brightness-contrast' as const }];
    const doc = makeDoc([groupContract('g', [a, groupContract('h', [b])]), c]);
    const { callbacks, set } = setup(doc);

    set({
      ...doc,
      stacks: stacksFrom([groupContract('g', [a, groupContract('h', [b])], { adjustments: stack }), c]),
    });
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onLayersRecomposite).toHaveBeenLastCalledWith(['a', 'b']);
    expect(callbacks.onLayerOrderChanged).not.toHaveBeenCalled();
  });
});

describe('createDocumentMirror: value edits through the reducer', () => {
  const setup = (doc: CanvasDocumentContractV3) => {
    let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
      document: doc,
      type: 'replaceCanvasDocument',
    });
    const store = createFakeStore([project]);
    const callbacks = spyCallbacks();
    const mirror = createDocumentMirror(store, project.id, callbacks);
    resetForestDiffCount();
    const apply = (mutation: CanvasProjectMutation, notify = true): void => {
      project = applyCanvasProjectMutation(project, mutation);
      (notify ? store.setState : store.replaceStateSilently)({ projects: [project] });
    };
    return { apply, callbacks, mirror };
  };

  it('reports leaf property and source edits from the recorded changes without diffing the forests', () => {
    const { apply, callbacks } = setup(makeDoc([rasterLayer('a'), rasterLayer('b')]));

    apply({ id: 'b', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['b'], [], false);
    apply({ id: 'a', source: { bitmap: null, type: 'paint' }, type: 'updateCanvasLayerSource' });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a'], ['a'], false);
    expect(getForestDiffCount()).toBe(0);
  });

  it('folds value edits a missed notification skipped, and diffs forests only across a structural edit', () => {
    const { apply, callbacks, mirror } = setup(makeDoc([rasterLayer('a'), rasterLayer('b'), rasterLayer('c')]));

    apply({ id: 'a', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' }, false);
    apply({ id: 'c', source: { bitmap: null, type: 'paint' }, type: 'updateCanvasLayerSource' }, false);
    mirror.refresh();
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a', 'c'], ['c'], false);
    expect(getForestDiffCount()).toBe(0);

    apply({ ids: ['b'], type: 'removeCanvasLayers' });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['b'], [], true);
    expect(getForestDiffCount()).toBe(1);
  });
});

describe('createDocumentMirror: change signal', () => {
  it('signals each document change once, after the other callbacks reacted, and never for staging alone', () => {
    const doc = makeDoc([rasterLayer('a')]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const order: string[] = [];
    const callbacks = {
      ...spyCallbacks(),
      onDocumentChanged: vi.fn(() => order.push('document')),
      onLayersChanged: vi.fn(() => order.push('layers')),
    };
    createDocumentMirror(store, 'p1', callbacks);

    store.setState({
      projects: [{ canvas: { ...canvas, document: makeDoc([rasterLayer('a', { opacity: 0.5 })]) }, id: 'p1' }],
    });
    store.setState({
      projects: [{ canvas: { ...store.getState().projects[0]!.canvas, stagingArea: makeStaging() }, id: 'p1' }],
    });

    expect(order).toEqual(['layers', 'document']);
  });
});
