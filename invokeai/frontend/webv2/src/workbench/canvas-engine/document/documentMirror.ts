import type {
  CanvasDocumentContractV2,
  CanvasStagingAreaContractV2,
  CanvasStateContractV2,
} from '@workbench/canvas-engine/contracts';

/** The minimal store shape the mirror depends on (a superset of `WorkbenchStore`). */
export interface DocumentMirrorStore {
  getCanvasState?(): CanvasStateContractV2 | null;
  getState?(): { projects: readonly { id: string; canvas: CanvasStateContractV2 }[] };
  subscribe(listener: () => void): () => void;
}

/** Callbacks fired when the mirrored document changes. */
export interface DocumentMirrorCallbacks {
  /**
   * One or more layers were added, removed, or replaced (the `changed` ids).
   * `sourceChanged` is the subset whose rasterization source (`source` for
   * raster/control layers, `mask` for guidance/mask layers) reference changed,
   * plus any newly added layer — i.e. the layers whose cached pixels are now
   * stale. A prop/transform-only edit reports the id in `changed` but NOT in
   * `sourceChanged`, so the engine keeps (rather than clears) its raster cache.
   */
  onLayersChanged(changed: string[], sourceChanged: string[]): void;
  /**
   * The layers array was reordered (a new array reference, same element
   * references, different sequence) — recomposite the document with the new
   * z-order, but nothing needs re-rasterizing.
   */
  onLayerOrderChanged(): void;
  /** The document was replaced wholesale (dims/background change, appear/disappear) — full invalidate. */
  onDocumentReplaced(): void;
  /** The generation bounding box changed. */
  onBboxChanged(): void;
  /** The staging area changed. */
  onStagingChanged(): void;
}

/** The imperative mirror handle. */
export interface DocumentMirror {
  /** The current mirrored document, or `null` if the project is gone. */
  getDocument(): CanvasDocumentContractV2 | null;
  /** Synchronously reconciles from store state when ordinary notification was interrupted. */
  refresh(): void;
  /** Removes the store subscription. */
  dispose(): void;
}

type Bbox = CanvasDocumentContractV2['bbox'];

const bboxEqual = (a: Bbox, b: Bbox): boolean =>
  a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

/**
 * The reference whose change requires re-rasterizing a layer's cache: the
 * `source` for raster/control layers, the mask's `bitmap` for guidance/mask
 * layers. The reducer preserves this reference across a prop/transform-only edit
 * (it spreads `...layer`), so comparing it distinguishes a genuine source swap
 * from an opacity/blend/lock/visibility/rename/nudge tweak.
 *
 * For masks the reference is the mask BITMAP, not the whole `mask` object: the
 * cache holds only the alpha stencil, so a fill-only change (colour/style/noise/
 * denoise-limit) must NOT invalidate it. Invalidating on a fill tweak would
 * re-rasterize from `mask.bitmap` — clearing unflushed brush strokes that live
 * only in the cache — while the compositor already reads the fresh fill from the
 * mirror at draw time. A genuine bitmap swap (persistence round-trip, undo,
 * import) still changes this reference and re-rasterizes (guarded by self-echo).
 */
const rasterSourceRef = (layer: CanvasDocumentContractV2['layers'][number]): unknown =>
  layer.type === 'raster' || layer.type === 'control' ? layer.source : layer.mask.bitmap;

/**
 * Diffs two layer arrays by object identity, keyed by layer id. Returns the ids
 * of layers that were added, removed, or replaced (`changed`), and the subset
 * whose rasterization source reference changed or that were newly added
 * (`sourceChanged`) — the layers whose cached pixels are now stale.
 */
const diffLayers = (
  prev: readonly CanvasDocumentContractV2['layers'][number][],
  next: readonly CanvasDocumentContractV2['layers'][number][]
): { changed: string[]; sourceChanged: string[] } => {
  const prevById = new Map(prev.map((layer) => [layer.id, layer]));
  const nextById = new Map(next.map((layer) => [layer.id, layer]));
  const changed = new Set<string>();
  const sourceChanged = new Set<string>();

  for (const layer of next) {
    const before = prevById.get(layer.id);
    if (!before) {
      // Added: no prior cache, so its source is new by definition.
      changed.add(layer.id);
      sourceChanged.add(layer.id);
    } else if (before !== layer) {
      changed.add(layer.id);
      if (rasterSourceRef(before) !== rasterSourceRef(layer)) {
        sourceChanged.add(layer.id);
      }
    }
  }
  for (const layer of prev) {
    if (!nextById.has(layer.id)) {
      // Removed: reported so the engine drops its cache. Not a source change.
      changed.add(layer.id);
    }
  }
  return { changed: [...changed], sourceChanged: [...sourceChanged] };
};

/**
 * True when two layer arrays hold the same ids in a different sequence.
 * Only meaningful to call once `diffLayers` has already confirmed no id was
 * added, removed, or replaced by reference.
 */
const layerOrderChanged = (
  prev: readonly CanvasDocumentContractV2['layers'][number][],
  next: readonly CanvasDocumentContractV2['layers'][number][]
): boolean => {
  if (prev.length !== next.length) {
    return true;
  }
  for (let i = 0; i < prev.length; i++) {
    if (prev[i]?.id !== next[i]?.id) {
      return true;
    }
  }
  return false;
};

/**
 * Creates a document mirror bound to `projectId`. Subscribes immediately and
 * seeds the last-seen references from the current state (so no spurious
 * callback fires on creation).
 */
export const createDocumentMirror = (
  store: DocumentMirrorStore,
  projectIdOrCallbacks: string | DocumentMirrorCallbacks,
  maybeCallbacks?: DocumentMirrorCallbacks
): DocumentMirror => {
  const projectId = typeof projectIdOrCallbacks === 'string' ? projectIdOrCallbacks : null;
  const callbacks = typeof projectIdOrCallbacks === 'string' ? maybeCallbacks : projectIdOrCallbacks;
  if (!callbacks) {
    throw new Error('DocumentMirror callbacks are required.');
  }
  const selectCanvas = (): CanvasStateContractV2 | null =>
    store.getCanvasState?.() ??
    (projectId === null
      ? null
      : (store.getState?.().projects.find((project) => project.id === projectId)?.canvas ?? null));

  let lastDoc: CanvasDocumentContractV2 | null = selectCanvas()?.document ?? null;
  let lastRevision: number = selectCanvas()?.documentRevision ?? 0;
  let lastStaging: CanvasStagingAreaContractV2 | null = selectCanvas()?.stagingArea ?? null;

  const handleChange = (): void => {
    const canvas = selectCanvas();
    const doc = canvas?.document ?? null;
    const revision = canvas?.documentRevision ?? 0;
    const staging = canvas?.stagingArea ?? null;

    if (doc !== lastDoc) {
      const prevDoc = lastDoc;
      const prevRevision = lastRevision;
      lastDoc = doc;
      lastRevision = revision;

      if (!prevDoc || !doc) {
        // Document appeared or disappeared: treat as a full replacement.
        callbacks.onDocumentReplaced();
      } else if (
        revision !== prevRevision ||
        prevDoc.width !== doc.width ||
        prevDoc.height !== doc.height ||
        prevDoc.background !== doc.background
      ) {
        // A wholesale swap (revision bump) or a dims/background change: the pixel
        // history no longer describes the live document.
        callbacks.onDocumentReplaced();
      } else {
        if (prevDoc.layers !== doc.layers) {
          const { changed, sourceChanged } = diffLayers(prevDoc.layers, doc.layers);
          if (changed.length > 0) {
            callbacks.onLayersChanged(changed, sourceChanged);
          } else if (layerOrderChanged(prevDoc.layers, doc.layers)) {
            callbacks.onLayerOrderChanged();
          }
        }
        if (!bboxEqual(prevDoc.bbox, doc.bbox)) {
          callbacks.onBboxChanged();
        }
      }
    }

    if (staging !== lastStaging) {
      lastStaging = staging;
      callbacks.onStagingChanged();
    }
  };

  const unsubscribe = store.subscribe(handleChange);

  return {
    dispose: unsubscribe,
    getDocument: () => lastDoc,
    refresh: handleChange,
  };
};
