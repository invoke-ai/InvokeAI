import type {
  CanvasDocumentContractV2,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStagingAreaContractV2,
  CanvasStateContractV2,
} from '@workbench/types';

import { describe, expect, it, vi } from 'vitest';

import type { DocumentMirrorCallbacks } from './documentMirror';

import { createDocumentMirror } from './documentMirror';

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
  layers: CanvasLayerContract[],
  overrides: Partial<CanvasDocumentContractV2> = {}
): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId: null,
  version: 2,
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

const makeCanvas = (document: CanvasDocumentContractV2, documentRevision = 0): CanvasStateContractV2 => ({
  document,
  documentRevision,
  snapshots: [],
  stagingArea: makeStaging(),
  version: 2,
});

interface FakeProject {
  id: string;
  canvas: CanvasStateContractV2;
}

const createFakeStore = (projects: FakeProject[]) => {
  let state = { projects };
  const listeners = new Set<() => void>();
  return {
    getState: () => state,
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
  onStagingChanged: vi.fn<DocumentMirrorCallbacks['onStagingChanged']>(),
});

describe('createDocumentMirror', () => {
  it('reports exactly the edited layer id when one layer changes by reference', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const doc = makeDoc([a, b]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    // Replace only layer `a` (new object), keep `b` identity. A prop-only edit
    // (like the reducer's `updateCanvasLayer`) keeps the `source` reference, so
    // the id is reported as changed but NOT source-changed.
    const nextDoc: CanvasDocumentContractV2 = { ...doc, layers: [{ ...a, opacity: 0.5 }, b] };
    store.setState({ projects: [{ canvas: { ...canvas, document: nextDoc }, id: 'p1' }] });

    expect(callbacks.onLayersChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).toHaveBeenCalledWith(['a'], []);
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

    // Prop-only edit (opacity): spreading the prior layer preserves its `source`
    // reference exactly as the reducer does, so the engine must NOT re-rasterize
    // (which would clear an unflushed paint layer).
    const opacityEdit: CanvasDocumentContractV2 = { ...doc, layers: [{ ...a, opacity: 0.5 }] };
    store.setState({ projects: [{ canvas: { ...canvas, document: opacityEdit }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a'], []);

    // Genuine source swap (new `source` object): reported as source-changed.
    const swapped = store.getState().projects[0]!.canvas.document.layers[0] as CanvasRasterLayerContractV2;
    const sourceSwap: CanvasDocumentContractV2 = {
      ...doc,
      layers: [{ ...swapped, source: { image: { height: 10, imageName: 'a-v2', width: 10 }, type: 'image' } }],
    };
    store.setState({ projects: [{ canvas: { ...canvas, document: sourceSwap }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a'], ['a']);
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

    // A fill-only change (new `mask` object, same `bitmap` ref): reported as
    // changed but NOT source-changed — invalidating would clear unflushed strokes.
    const fillEdit: CanvasDocumentContractV2 = {
      ...doc,
      layers: [{ ...mask, mask: { bitmap: null, fill: { color: '#00ff00', style: 'grid' } } }],
    };
    store.setState({ projects: [{ canvas: { ...canvas, document: fillEdit }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['m'], []);

    // A bitmap swap (persistence round-trip / undo): reported as source-changed.
    const current = store.getState().projects[0]!.canvas.document.layers[0] as CanvasInpaintMaskLayerContract;
    const bitmapSwap: CanvasDocumentContractV2 = {
      ...doc,
      layers: [{ ...current, mask: { ...current.mask, bitmap: { height: 20, imageName: 'mask-v1', width: 30 } } }],
    };
    store.setState({ projects: [{ canvas: { ...canvas, document: bitmapSwap }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['m'], ['m']);
  });

  it('reports added and removed layer ids', () => {
    const a = rasterLayer('a');
    const doc = makeDoc([a]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const b = rasterLayer('b');
    // Added layers are reported as source-changed (no prior cache to keep).
    store.setState({ projects: [{ canvas: { ...canvas, document: { ...doc, layers: [b, a] } }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['b'], ['b']);

    // A removal is a change but not a source change (the id has no incoming source).
    store.setState({ projects: [{ canvas: { ...canvas, document: { ...doc, layers: [b] } }, id: 'p1' }] });
    expect(callbacks.onLayersChanged).toHaveBeenLastCalledWith(['a'], []);
  });

  it('fires onLayerOrderChanged exactly once on a pure reorder, with no layer ids reported', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const doc = makeDoc([a, b]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    // New array reference, same element references, swapped order.
    store.setState({ projects: [{ canvas: { ...canvas, document: { ...doc, layers: [b, a] } }, id: 'p1' }] });

    expect(callbacks.onLayerOrderChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
    expect(callbacks.onBboxChanged).not.toHaveBeenCalled();
  });

  it('does not fire onLayerOrderChanged when the layers array is replaced with an identical order', () => {
    const a = rasterLayer('a');
    const b = rasterLayer('b');
    const doc = makeDoc([a, b]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    // New array reference, same element references, same order: a true no-op churn.
    store.setState({ projects: [{ canvas: { ...canvas, document: { ...doc, layers: [a, b] } }, id: 'p1' }] });

    expect(callbacks.onLayerOrderChanged).not.toHaveBeenCalled();
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
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

    // A same-dims snapshot restore: structuredClone reuses layer ids and keeps
    // width/height/background, so only the revision bump signals the swap.
    const restored = structuredClone(doc);
    store.setState({ projects: [{ canvas: makeCanvas(restored, 1), id: 'p1' }] });

    expect(callbacks.onDocumentReplaced).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onLayerOrderChanged).not.toHaveBeenCalled();
  });

  it('does not fire onDocumentReplaced for an ordinary layer edit at an unchanged revision', () => {
    const a = rasterLayer('a');
    const doc = makeDoc([a]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    const nextDoc: CanvasDocumentContractV2 = { ...doc, layers: [{ ...a, opacity: 0.5 }] };
    store.setState({ projects: [{ canvas: { ...canvas, document: nextDoc }, id: 'p1' }] });

    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
    expect(callbacks.onLayersChanged).toHaveBeenCalledWith(['a'], []);
  });

  it('fires onBboxChanged when only the bbox moves', () => {
    const layers = [rasterLayer('a')];
    const doc = makeDoc(layers);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    // Same layers array identity, new bbox.
    store.setState({
      projects: [
        { canvas: { ...canvas, document: { ...doc, bbox: { height: 100, width: 100, x: 10, y: 10 } } }, id: 'p1' },
      ],
    });
    expect(callbacks.onBboxChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
  });

  it('fires onStagingChanged when the staging area reference changes', () => {
    const doc = makeDoc([rasterLayer('a')]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    createDocumentMirror(store, 'p1', callbacks);

    store.setState({ projects: [{ canvas: { ...canvas, stagingArea: makeStaging() }, id: 'p1' }] });
    expect(callbacks.onStagingChanged).toHaveBeenCalledTimes(1);
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
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

    // Mutate only p2; p1's canvas keeps its identity.
    store.setState({
      projects: [
        { canvas: canvasA, id: 'p1' },
        { canvas: makeCanvas(makeDoc([rasterLayer('z', { opacity: 0.2 })])), id: 'p2' },
      ],
    });
    expect(callbacks.onLayersChanged).not.toHaveBeenCalled();
    expect(callbacks.onDocumentReplaced).not.toHaveBeenCalled();
    expect(callbacks.onBboxChanged).not.toHaveBeenCalled();
    expect(callbacks.onStagingChanged).not.toHaveBeenCalled();
  });

  it('reports null and stays no-op safe when the project is deleted', () => {
    const doc = makeDoc([rasterLayer('a')]);
    const canvas = makeCanvas(doc);
    const store = createFakeStore([{ canvas, id: 'p1' }]);
    const callbacks = spyCallbacks();
    const mirror = createDocumentMirror(store, 'p1', callbacks);

    store.setState({ projects: [] });
    expect(callbacks.onDocumentReplaced).toHaveBeenCalledTimes(1);
    expect(mirror.getDocument()).toBeNull();

    // Further unrelated churn: no additional callbacks.
    store.setState({ projects: [] });
    expect(callbacks.onDocumentReplaced).toHaveBeenCalledTimes(1);
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
