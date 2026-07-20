/**
 * The composite-generation operation: one deep call that turns the live canvas
 * into every uploaded image a canvas invoke needs.
 *
 * `composeForGeneration` owns the whole snapshot → plan → capture → composite
 * protocol that callers previously drove step by step through
 * `CanvasCompositeTransaction`: it captures the document snapshot, plans the
 * base / mask / control / regional composites, detaches the required layer
 * surfaces, runs a bounds-only pre-pass (content that does not overlap the bbox
 * is txt2img — no base composite, no upload, and the mode strategy is never
 * consulted), executes the base composite + grayscale masks + per-layer control
 * and regional-mask composites, and releases the raster snapshot in a `finally`
 * on both success and failure. Pixels, surfaces, dedupe caches, and memory
 * reservations never cross the seam.
 *
 * Caller-owned policy enters as three strategies on the options:
 * - `detectMode` maps composite facts to a generation mode. It is called at
 *   most once, and only when raster content overlaps the bbox.
 * - `shouldCompositeControlLayer` vets each planned control layer in z-order:
 *   `false` skips the layer (no upload); a **throw aborts the whole operation**
 *   (the snapshot is released, no dedupe entry is committed, and the error is
 *   rethrown to the caller).
 * - `shouldCompositeRegionalMask` vets each planned regional-guidance layer in
 *   z-order: `false` silently skips the region (no upload).
 *
 * Successful composition returns one opaque, idempotent dedupe commit. The
 * caller publishes it only after graph compilation and synchronous queue
 * dispatch both succeed; every earlier failure discards the operation-local
 * cache while the uploaded intermediates remain safe to retry.
 */

import type {
  CanvasControlLayerContract,
  CanvasDocumentSnapshot,
  CanvasRegionalGuidanceLayerContract,
  CanvasStateContractV2,
} from '@workbench/canvas-engine/api';
import type { CaptureRasterSnapshotResult } from '@workbench/canvas-engine/rasterTransactions';
import type { Rect } from '@workbench/canvas-engine/types';
import type {
  CompositeDedupeCache,
  ExecuteCompositePlanDeps,
} from '@workbench/canvas-operations/compositeForGeneration';

import { intersect } from '@workbench/canvas-engine/math/rect';
import {
  computeCompositeContentBounds,
  executeCompositePlan,
  executeControlComposite,
  executeMaskComposite,
  executeRegionalMaskComposite,
} from '@workbench/canvas-operations/compositeForGeneration';
import {
  planComposites,
  planControlComposites,
  planRegionalMaskComposites,
} from '@workbench/canvas-operations/generationCompositePlan';

/**
 * The generation mode a composite resolves to. Structural mirror of
 * generation's `CanvasGenerationMode` — canvas-operations must not import the
 * feature, and tsc guards drift in both directions at the caller's `detectMode`
 * wiring.
 */
export type GenerationCompositeMode = 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';

/** The composite facts the caller's `detectMode` strategy decides from. */
export interface GenerationModeFacts {
  /** The generation bounding box, in document space. */
  bbox: Rect;
  /** Union of enabled raster content bounds in document space, or `null`. */
  contentBounds: Rect | null;
  /** Whether the composited bbox surface is fully opaque (no transparent holes). */
  bboxFullyCovered: boolean;
  /** Whether an enabled inpaint mask with masked (non-white) content exists. */
  hasActiveInpaintMask: boolean;
}

/** Caller-owned policy + cancellation for {@link composeForGeneration}. */
export interface ComposeForGenerationOptions {
  /** Cancels raster capture; a capture abort resolves as `{ status: 'aborted' }`. */
  signal: AbortSignal;
  /**
   * Maps composite facts to the generation mode. Called at most once, and only
   * when raster content overlaps the bbox (otherwise the mode is txt2img).
   */
  detectMode(facts: GenerationModeFacts): GenerationCompositeMode;
  /**
   * Vets one planned control layer (called in z-order). `false` skips it with
   * no upload; a throw aborts the whole operation (snapshot released, dedupe
   * uncommitted, error rethrown). Omitted → every planned layer composites.
   */
  shouldCompositeControlLayer?(layer: CanvasControlLayerContract): boolean;
  /**
   * Vets one planned regional-guidance layer (called in z-order). `false`
   * silently skips it with no upload. Omitted → every planned region composites.
   */
  shouldCompositeRegionalMask?(layer: CanvasRegionalGuidanceLayerContract): boolean;
}

/** The engine-side executor dependencies the host supplies (dedupe + surfaces are operation-owned). */
export type GenerationCompositeExecutorDeps = Omit<ExecuteCompositePlanDeps, 'dedupe' | 'getLayerSurface'>;

/**
 * The engine seam {@link composeForGeneration} runs against. Production wiring
 * lives in `createCanvasEngine`; tests assemble a fake host over
 * `render/raster.testStub` + a mock uploader.
 */
export interface GenerationCompositeHost {
  /** Captures the current document snapshot, or `null` without an active document. */
  captureDocumentSnapshot(): CanvasDocumentSnapshot | null;
  /** Detaches the listed layers' pixel surfaces for the snapshot (status on failure). */
  captureRasterSnapshot(
    snapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options: { signal: AbortSignal }
  ): Promise<CaptureRasterSnapshotResult>;
  /** The engine's composite executor deps (backend, upload, reserve, ...). */
  getCompositeExecutorDeps(): GenerationCompositeExecutorDeps;
  /** Engine-scoped dedupe cache, merged into only by the returned commit. */
  dedupe: CompositeDedupeCache;
}

declare const generationCompositeDedupeCommitBrand: unique symbol;

/** Opaque publication handle; callers may only commit the completed transaction. */
export interface GenerationCompositeDedupeCommit {
  readonly [generationCompositeDedupeCommitBrand]: true;
  commit(): void;
}

/** Everything a canvas invoke needs from the composite pipeline. */
export interface GenerationComposites {
  /** The frozen canvas state the composites were built from (for the enqueue snapshot). */
  canvas: CanvasStateContractV2;
  /** The generation bounding box of the frozen document. */
  bbox: Rect;
  /** The resolved generation mode. */
  mode: GenerationCompositeMode;
  /** The base composite's uploaded image, or `null` for txt2img. */
  baseImageName: string | null;
  /** The grayscale inpaint mask's uploaded image (inpaint/outpaint with mask content), or `null`. */
  maskImageName: string | null;
  /** The grayscale noise mask's uploaded image (inpaint/outpaint only), or `null`. */
  noiseMaskImageName: string | null;
  /** One uploaded image per accepted control layer, in z-order. */
  controlImages: { layerId: string; imageName: string }[];
  /** One uploaded alpha-mask image per accepted regional-guidance layer, in z-order. */
  regionalMaskImages: { layerId: string; imageName: string }[];
}

export type ComposeForGenerationResult =
  | { status: 'ok'; composites: GenerationComposites; dedupeCommit: GenerationCompositeDedupeCommit }
  | { status: 'no-document' | 'stale' | 'aborted' | 'not-ready' | 'over-budget' };

/**
 * Runs the complete canvas → generation composite pipeline against `host`.
 * Capture failures resolve as a status; execution failures (including a
 * control-predicate throw) release the raster snapshot and rethrow.
 */
export const composeForGeneration = async (
  host: GenerationCompositeHost,
  options: ComposeForGenerationOptions
): Promise<ComposeForGenerationResult> => {
  const documentSnapshot = host.captureDocumentSnapshot();
  if (!documentSnapshot) {
    return { status: 'no-document' };
  }
  const document = documentSnapshot.canvas.document;
  const bbox = document.bbox;

  const plan = planComposites(document, bbox);
  const controlPlan = planControlComposites(document, bbox);
  const regionalPlan = planRegionalMaskComposites(document, bbox);

  const requiredLayerIds = new Set<string>();
  for (const entry of [
    ...plan.entries,
    ...controlPlan.map((item) => item.entry),
    ...regionalPlan.map((item) => item.entry),
  ]) {
    for (const layer of entry.layers) {
      requiredLayerIds.add(layer.id);
    }
    for (const layer of entry.maskLayers ?? []) {
      requiredLayerIds.add(layer.id);
    }
  }

  const capture = await host.captureRasterSnapshot(documentSnapshot, [...requiredLayerIds], {
    signal: options.signal,
  });
  if (capture.status !== 'ok') {
    return capture;
  }
  const rasterSnapshot = capture.snapshot;

  // Operation-scoped dedupe: composites read/write a copy. Publication remains
  // provisional until the caller successfully compiles and dispatches.
  const operationDedupe: CompositeDedupeCache = {
    byHash: new Map(host.dedupe.byHash),
    byKey: new Map(host.dedupe.byKey),
  };
  const deps: ExecuteCompositePlanDeps = {
    ...host.getCompositeExecutorDeps(),
    dedupe: operationDedupe,
    getLayerSurface: (layerId) => {
      const detached = rasterSnapshot.layerSurfaces.get(layerId);
      return detached
        ? Promise.resolve(detached)
        : Promise.reject(new Error(`Canvas raster snapshot is missing layer ${layerId}.`));
    },
  };

  try {
    // Bounds-only pre-pass (pure geometry, no upload): content that does not
    // overlap the bbox is txt2img no matter the coverage. Also naturally skips a
    // zero-area bbox (`intersect` is null for empty rects). `intersect`'s strict
    // overlap matches generation's `rectsIntersect` (flush edges don't count).
    const contentBounds = computeCompositeContentBounds(plan);

    let mode: GenerationCompositeMode = 'txt2img';
    let baseImageName: string | null = null;
    let maskImageName: string | null = null;
    let noiseMaskImageName: string | null = null;

    if (contentBounds && intersect(contentBounds, bbox) !== null) {
      const result = await executeCompositePlan(plan, deps);
      baseImageName = result.base.imageName;

      // The grayscale denoise-limit mask (when enabled inpaint masks exist) both
      // decides inpaint-vs-img2img (its coverage) and feeds the graph — it must
      // execute BEFORE the mode strategy runs. Legacy parity: raster opaque +
      // mask has content → inpaint; else img2img.
      const maskEntry = plan.entries.find((entry) => entry.kind === 'inpaint-mask');
      const maskResult = maskEntry ? await executeMaskComposite(maskEntry, deps) : null;

      mode = options.detectMode({
        bbox,
        bboxFullyCovered: result.bboxFullyCovered,
        contentBounds: result.contentBounds,
        hasActiveInpaintMask: maskResult?.hasContent ?? false,
      });

      if (mode === 'inpaint' || mode === 'outpaint') {
        // A real (non-white) mask feeds create_gradient_mask; outpaint without one
        // derives its mask from the raster alpha (maskImageName stays null).
        maskImageName = maskResult?.hasContent ? maskResult.imageName : null;
        const noiseEntry = plan.entries.find((entry) => entry.kind === 'noise-mask');
        if (noiseEntry) {
          noiseMaskImageName = (await executeMaskComposite(noiseEntry, deps)).imageName;
        }
      }
    }

    const layerById = new Map(document.layers.map((layer) => [layer.id, layer]));

    // Control layers apply in every mode, independent of the base composite —
    // composite each accepted layer regardless of whether raster content
    // overlaps. Each layer is composited SEPARATELY (never blended).
    const controlImages: { layerId: string; imageName: string }[] = [];
    for (const { entry, layerId } of controlPlan) {
      const layer = layerById.get(layerId);
      if (!layer || layer.type !== 'control') {
        continue;
      }
      if (options.shouldCompositeControlLayer && !options.shouldCompositeControlLayer(layer)) {
        continue;
      }
      const result = await executeControlComposite(entry, deps);
      controlImages.push({ imageName: result.imageName, layerId });
    }

    // Regional guidance also applies in every mode — each accepted region's
    // mask is composited separately (its alpha feeds its own tensor).
    const regionalMaskImages: { layerId: string; imageName: string }[] = [];
    for (const { entry, layerId } of regionalPlan) {
      const layer = layerById.get(layerId);
      if (!layer || layer.type !== 'regional_guidance') {
        continue;
      }
      if (options.shouldCompositeRegionalMask && !options.shouldCompositeRegionalMask(layer)) {
        continue;
      }
      const result = await executeRegionalMaskComposite(entry, deps);
      regionalMaskImages.push({ imageName: result.imageName, layerId });
    }

    let isCommitted = false;
    const dedupeCommit = {
      commit: (): void => {
        if (isCommitted) {
          return;
        }
        isCommitted = true;
        for (const [key, value] of operationDedupe.byHash) {
          host.dedupe.byHash.set(key, value);
        }
        for (const [key, value] of operationDedupe.byKey) {
          host.dedupe.byKey.set(key, value);
        }
      },
    } as GenerationCompositeDedupeCommit;

    return {
      composites: {
        baseImageName,
        bbox,
        canvas: rasterSnapshot.canvas,
        controlImages,
        maskImageName,
        mode,
        noiseMaskImageName,
        regionalMaskImages,
      },
      dedupeCommit,
      status: 'ok',
    };
  } finally {
    rasterSnapshot.release();
  }
};
