/** Always release captured rasters; the returned dedupe commit publishes only after compile and dispatch succeed. */

import type {
  CanvasControlLayerContract,
  CanvasDocumentSnapshot,
  CanvasInpaintMaskLayerContract,
  CanvasRegionalGuidanceLayerContract,
  CanvasStateContractV3,
  LayerUnreadable,
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
  type AdjustmentRefusal,
  AdjustmentRefusedError,
  adjustmentRefusal,
} from '@workbench/canvas-engine/render/exactAdjustments';
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

/** Mirror CanvasGenerationMode without importing the feature; caller wiring checks compatibility in both directions. */
export type GenerationCompositeMode = 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';

export interface GenerationModeFacts {
  /** The generation bounding box, in document space. */
  bbox: Rect;
  /** Union of enabled raster content bounds in document space, or `null`. */
  contentBounds: Rect | null;
  bboxFullyCovered: boolean;
  bboxHasContent: boolean;
  hasActiveInpaintMask: boolean;
}

export interface ComposeForGenerationOptions {
  /** Cancels raster capture; a capture abort resolves as `{ status: 'aborted' }`. */
  signal: AbortSignal;
  /** Called at most once, and only when raster content overlaps the bbox (otherwise txt2img). */
  detectMode(facts: GenerationModeFacts): GenerationCompositeMode;
  /** `false` skips the layer with no upload; a throw aborts the operation, dedupe uncommitted. */
  shouldCompositeControlLayer?(layer: CanvasControlLayerContract): boolean;
  shouldCompositeRegionalMask?(layer: CanvasRegionalGuidanceLayerContract): boolean;
  /** Asked only while the mask inpaints; absent means its guidance never contributes. */
  shouldCompositeMaskGuidance?(layer: CanvasInpaintMaskLayerContract): boolean;
}

export type GenerationCompositeExecutorDeps = Omit<ExecuteCompositePlanDeps, 'dedupe' | 'getLayerSurface'>;

export interface GenerationCompositeHost {
  captureDocumentSnapshot(): CanvasDocumentSnapshot | null;
  captureRasterSnapshot(
    snapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options: { signal: AbortSignal; region: Rect }
  ): Promise<CaptureRasterSnapshotResult>;
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

export interface GenerationComposites {
  canvas: CanvasStateContractV3;
  bbox: Rect;
  mode: GenerationCompositeMode;
  baseImageName: string | null;
  maskImageName: string | null;
  noiseMaskImageName: string | null;
  controlImages: { layerId: string; imageName: string }[];
  /** `bounds` in bbox space (origin at the bbox corner; empty when the mask misses the bbox). */
  regionalMaskImages: { layerId: string; imageName: string; bounds: Rect }[];
}

export type ComposeForGenerationResult =
  | { status: 'ok'; composites: GenerationComposites; dedupeCommit: GenerationCompositeDedupeCommit }
  | { status: 'no-document' | 'stale' | 'aborted' | 'not-ready' | 'over-budget' }
  | LayerUnreadable
  | AdjustmentRefusal;

/** Capture failures resolve as a status; execution failures release the raster snapshot and rethrow. */
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
    region: bbox,
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

  // Operation-scoped dedupe copy; publication waits for the caller to compile and dispatch.
  const operationDedupe: CompositeDedupeCache = {
    byHash: new Map(host.dedupe.byHash),
    byKey: new Map(host.dedupe.byKey),
  };
  const executorDeps = host.getCompositeExecutorDeps();
  const deps: ExecuteCompositePlanDeps = {
    ...executorDeps,
    dedupe: operationDedupe,
    signal: options.signal,
    getLayerSurface: (layerId) => {
      const detached = rasterSnapshot.layerSurfaces.get(layerId);
      return detached
        ? Promise.resolve({ ...detached, release: () => undefined })
        : Promise.reject(new Error(`Canvas raster snapshot is missing layer ${layerId}.`));
    },
    uploadImage: async (blob) => {
      // Re-check at the transport boundary so Account A work cannot start with Account B credentials.
      options.signal.throwIfAborted();
      const uploaded = await executorDeps.uploadImage(blob);
      options.signal.throwIfAborted();

      return uploaded;
    },
  };

  try {
    // Empty boxes and touching edges do not overlap and need no base upload.
    const actualLayerRects = new Map<string, Rect>();
    for (const [layerId, detached] of rasterSnapshot.layerSurfaces) {
      actualLayerRects.set(layerId, detached.sourceRect ?? detached.rect);
    }
    const contentBounds = computeCompositeContentBounds(plan, actualLayerRects);

    let mode: GenerationCompositeMode = 'txt2img';
    let baseImageName: string | null = null;
    let maskImageName: string | null = null;
    let noiseMaskImageName: string | null = null;

    if (contentBounds && intersect(contentBounds, bbox) !== null) {
      const result = await executeCompositePlan(plan, deps);

      // The mask comes before mode detection: opaque raster content becomes inpaint when the mask has coverage.
      const maskEntry = plan.entries.find((entry) => entry.kind === 'inpaint-mask');
      const maskResult = maskEntry ? await executeMaskComposite(maskEntry, deps) : null;

      mode = options.detectMode({
        bbox,
        bboxFullyCovered: result.bboxFullyCovered,
        bboxHasContent: result.bboxHasContent,
        contentBounds,
        hasActiveInpaintMask: maskResult?.hasContent ?? false,
      });
      baseImageName = mode === 'txt2img' ? null : result.base.imageName;

      if (mode === 'inpaint' || mode === 'outpaint') {
        // Outpaint without a real mask derives it from the raster alpha.
        maskImageName = maskResult?.hasContent ? maskResult.imageName : null;
        const noiseEntry = plan.entries.find((entry) => entry.kind === 'noise-mask');
        if (noiseEntry) {
          noiseMaskImageName = (await executeMaskComposite(noiseEntry, deps)).imageName;
        }
      }
    }

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

    const regionalMaskImages: GenerationComposites['regionalMaskImages'] = [];
    for (const { entry, layerId } of regionalPlan) {
      const layer = lookupDocumentLayer(document, layerId);
      if (layer?.type === 'regional_guidance') {
        if (options.shouldCompositeRegionalMask && !options.shouldCompositeRegionalMask(layer)) {
          continue;
        }
        const result = await executeRegionalMaskComposite(entry, deps);
        regionalMaskImages.push({ bounds: result.bounds, imageName: result.imageName, layerId });
      } else if (layer?.type === 'inpaint_mask') {
        // A mask and its modifiers are inert unless it inpaints, like the noise mask above.
        if ((mode !== 'inpaint' && mode !== 'outpaint') || !options.shouldCompositeMaskGuidance?.(layer)) {
          continue;
        }
        const result = await executeRegionalMaskComposite(entry, deps);
        // A region that misses the frame would still add conditioning and, with auto-negative, a whole-frame negative.
        if (result.bounds.width > 0 && result.bounds.height > 0) {
          regionalMaskImages.push({ bounds: result.bounds, imageName: result.imageName, layerId });
        }
      }
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
  } catch (error) {
    if (error instanceof AdjustmentRefusedError) {
      return adjustmentRefusal(document, error.ownerId, error.reason);
    }
    throw error;
  } finally {
    rasterSnapshot.release();
  }
};
