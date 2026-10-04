/**
 * Own snapshot, planning, capture, and all generation composites; always release captured rasters. No bbox overlap
 * means txt2img without base upload or mode detection. detectMode runs at most once; control/regional predicates
 * skip rejected layers, while thrown errors abort without committing dedupe. Return an idempotent dedupe commit
 * for publication only after graph compilation and synchronous queue dispatch succeed.
 */

import type {
  CanvasControlLayerContract,
  CanvasDocumentSnapshot,
  CanvasRegionalGuidanceLayerContract,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/api';
import type { CaptureRasterSnapshotResult } from '@workbench/canvas-engine/rasterTransactions';
import type { Rect } from '@workbench/canvas-engine/types';
import type {
  CompositeDedupeCache,
  ExecuteCompositePlanDeps,
} from '@workbench/canvas-operations/compositeForGeneration';

import { lookupDocumentLayer } from '@workbench/canvas-engine/document-model/documentModel';
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
 * Mirror CanvasGenerationMode without importing the feature; caller wiring checks compatibility in both
 * directions.
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
  canvas: CanvasStateContractV3;
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
  if (options.signal.aborted) {
    return { status: 'aborted' };
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

  // Release captures returned after abort, preserving the no-pixel-output guarantee.
  if (options.signal.aborted) {
    rasterSnapshot.release();
    return { status: 'aborted' };
  }

  // Operation-scoped dedupe: composites read/write a copy. Publication remains
  // provisional until the caller successfully compiles and dispatches.
  const operationDedupe: CompositeDedupeCache = {
    byHash: new Map(host.dedupe.byHash),
    byKey: new Map(host.dedupe.byKey),
  };
  const executorDeps = host.getCompositeExecutorDeps();
  const deps: ExecuteCompositePlanDeps = {
    ...executorDeps,
    dedupe: operationDedupe,
    getLayerSurface: (layerId) => {
      const detached = rasterSnapshot.layerSurfaces.get(layerId);
      // The raster snapshot owns these detached pixels until the whole operation releases it.
      return detached
        ? Promise.resolve({ ...detached, release: () => undefined })
        : Promise.reject(new Error(`Canvas raster snapshot is missing layer ${layerId}.`));
    },
    uploadImage: async (blob) => {
      // Encoding and hashing yield before upload. Re-check at the transport
      // boundary so Account A work cannot start with Account B credentials.
      options.signal.throwIfAborted();
      const uploaded = await executorDeps.uploadImage(blob);
      options.signal.throwIfAborted();

      return uploaded;
    },
  };

  try {
    // Use captured rects, including measured text, for the bounds-only overlap check. Empty boxes and touching
    // edges do not overlap and need no base upload.
    const actualLayerRects = new Map<string, Rect>();
    for (const [layerId, detached] of rasterSnapshot.layerSurfaces) {
      actualLayerRects.set(layerId, detached.rect);
    }
    const contentBounds = computeCompositeContentBounds(plan, actualLayerRects);

    let mode: GenerationCompositeMode = 'txt2img';
    let baseImageName: string | null = null;
    let maskImageName: string | null = null;
    let noiseMaskImageName: string | null = null;

    if (contentBounds && intersect(contentBounds, bbox) !== null) {
      const result = await executeCompositePlan(plan, deps);
      baseImageName = result.base.imageName;

      // Composite the denoise-limit mask before mode detection: even opaque raster content becomes inpaint when
      // the mask has coverage.
      const maskEntry = plan.entries.find((entry) => entry.kind === 'inpaint-mask');
      const maskResult = maskEntry ? await executeMaskComposite(maskEntry, deps) : null;

      mode = options.detectMode({
        bbox,
        bboxFullyCovered: result.bboxFullyCovered,
        contentBounds,
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

    // Composite each accepted control separately in every mode, regardless of base-raster overlap.
    const controlImages: { layerId: string; imageName: string }[] = [];
    for (const { entry, layerId } of controlPlan) {
      const layer = lookupDocumentLayer(document, layerId);
      if (!layer || layer.type !== 'control') {
        continue;
      }
      if (options.shouldCompositeControlLayer && !options.shouldCompositeControlLayer(layer)) {
        continue;
      }
      const result = await executeControlComposite(entry, deps);
      controlImages.push({ imageName: result.imageName, layerId });
    }

    // Composite each accepted regional alpha mask separately in every mode.
    const regionalMaskImages: { layerId: string; imageName: string }[] = [];
    for (const { entry, layerId } of regionalPlan) {
      const layer = lookupDocumentLayer(document, layerId);
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
