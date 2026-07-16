/**
 * Wires the canvas-generation pipeline into the Invoke button.
 *
 * `prepareCanvasInvocation` is the thin command-layer entry: it resolves the
 * active project's engine, scopes a per-engine dedupe cache, and delegates to
 * the React-free {@link runCanvasInvocation} orchestrator. The orchestrator runs
 * the whole flush → composite → mode-detect → compile → enqueue flow with every
 * side-effecting dependency injected, so it is fully node-testable with fakes.
 *
 * ## Flow (per the plan)
 * 1. Resolve the engine + its live document (no engine/document → notice, abort).
 * 2. `flushPendingUploads()` — the paint-bitmap persistence barrier — so the
 *    composite reads the latest painted pixels.
 * 3. Plan the composites, then run a **bounds-only pre-pass**: if enabled raster
 *    content does not overlap the bbox, the mode is txt2img regardless of
 *    coverage, so we skip the composite/encode/upload entirely (no useless base
 *    image). Only when content overlaps do we run the executor (which the
 *    image-referencing modes need) and detect img2img/outpaint from coverage.
 * 4. Compile the base-appropriate graph and dispatch it to canvas staging.
 * 5. Any failure (validation throw, upload error, unsupported mode) records a
 *    notice — the app never gets stuck.
 *
 * A module-scoped in-flight guard drops an invoke while a prior prepare for the
 * same project is still running (the Invoke hotkey can be mashed).
 */

import type {
  CanvasDocumentSnapshot,
  CanvasDocumentCapability,
  CanvasExportCapability,
  CanvasLifecycleCapability,
  CaptureRasterSnapshotResult,
} from '@workbench/canvas-engine/api';
import type {
  CompositeDedupeCache,
  ExecuteCompositePlanDeps,
} from '@workbench/canvas-operations/compositeForGeneration';
import type { ControlLayerGraphInput } from '@workbench/generation/canvas/addControlLayers';
import type {
  RegionalGuidanceInput,
  RegionalReferenceImageInput,
} from '@workbench/generation/canvas/addRegionalGuidance';
import type { CanvasCompileMode } from '@workbench/generation/canvas/types';
import type { GenerateModelConfig, GenerateReferenceImage } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';
import type {
  CanvasDocumentContractV2,
  ProjectSettings,
  ResultDestination,
  WorkbenchNotificationKind,
} from '@workbench/types';
import type { CanvasCompositingSettings } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import type { WorkbenchAction } from '@workbench/workbenchState';

import {
  computeCompositeContentBounds,
  executeCompositePlan,
  executeControlComposite,
  executeMaskComposite,
  executeRegionalMaskComposite,
} from '@workbench/canvas-operations/compositeForGeneration';
import { getEngine } from '@workbench/canvas-operations/engineRegistry';
import {
  getRegionalGuidanceRejectionReason,
  isRegionalGuidanceSupportedForBase,
} from '@workbench/generation/canvas/addRegionalGuidance';
import { detectCanvasMode, rectsIntersect } from '@workbench/generation/canvas/canvasMode';
import { compileCanvasGraph } from '@workbench/generation/canvas/compileCanvasGraph';
import {
  planComposites,
  planControlComposites,
  planRegionalMaskComposites,
} from '@workbench/generation/canvas/compositePlan';
import { getControlValidationReason } from '@workbench/generation/canvas/controlValidation';
import { resolveGenerateSeed } from '@workbench/generation/graph';
import { normalizeGenerateWidgetValues, syncGenerateWidgetValuesWithModels } from '@workbench/generation/settings';
import { DEFAULT_CANVAS_COMPOSITING } from '@workbench/widgets/canvas/invoke/canvasCompositing';

/** Title on every canvas-invoke failure notice. */
export const CANVAS_INVOKE_ERROR_TITLE = 'Canvas generation failed';

/** Engine-backed executor deps, minus the caller-owned dedupe cache. */
type CompositeExecutorDeps = Omit<ExecuteCompositePlanDeps, 'dedupe' | 'getLayerSurface'>;

/** Injected dependencies for the React-free orchestrator. */
export interface RunCanvasInvocationDeps {
  /** The active project id (also the in-flight guard key). */
  projectId: string;
  /**
   * The resolved result destination. Drives the compiled output node's
   * `is_intermediate` flag (`destination === 'canvas'`) and the enqueued
   * snapshot's destination, so a Canvas source can target the Gallery.
   */
  destination: ResultDestination;
  /** Captures the exact post-flush canvas contract used by immutable invocation transactions. */
  captureDocumentSnapshot: () => CanvasDocumentSnapshot | null;
  /** Detaches every layer surface required by the transaction's complete composite plan. */
  captureRasterSnapshot: (
    snapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options: { signal: AbortSignal }
  ) => Promise<CaptureRasterSnapshotResult>;
  /** Cancels cache preparation and detachment for this invocation transaction. */
  signal: AbortSignal;
  /** Paint-bitmap persistence barrier, awaited before compositing. */
  flushPendingUploads: () => Promise<void>;
  /** Engine-backed composite executor deps (backend + rasterize-or-throw + uploader). */
  executorDeps: CompositeExecutorDeps;
  /** Per-engine dedupe cache (caller-owned; persists across invokes, size-capped). */
  dedupe: CompositeDedupeCache;
  /** Project ids with a prepare currently in flight (module/registry-scoped). */
  inFlight: Set<string>;
  /** The generate widget's raw persisted values (model/prompt/steps, shared with Generate). */
  generateValues: Record<string, unknown>;
  /** Loaded models, for the same value/model sync the generate path performs. */
  models?: readonly ModelConfig[];
  /** Project settings (only `useCpuNoise` is consulted by the compiler). */
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>;
  /** Persisted denoising strength (already defaulted + clamped). Used for every image mode. */
  strength: number;
  /** Persisted compositing settings (infill / coherence / mask blur), defaulted + clamped. */
  compositing: CanvasCompositingSettings;
  /** Reducer dispatch. */
  dispatch: (action: WorkbenchAction) => void;
}

const recordNotice = (
  dispatch: (action: WorkbenchAction) => void,
  kind: WorkbenchNotificationKind,
  message: string
): void => {
  dispatch({ kind, message, title: CANVAS_INVOKE_ERROR_TITLE, type: 'recordNotice' });
};

/**
 * Composites + resolves the enabled control layers into graph inputs. Each
 * content-bearing control layer is composited SEPARATELY (never blended) and its
 * adapter model resolved from the loaded models. Enabled nonempty invalid layers
 * throw a shared reason code and block invocation; content-less control layers
 * are already excluded by `planControlComposites` and remain harmless.
 */
const collectControlLayerInputs = async (
  document: CanvasDocumentContractV2,
  bbox: { x: number; y: number; width: number; height: number },
  model: GenerateModelConfig,
  models: readonly ModelConfig[] | undefined,
  executorDeps: ExecuteCompositePlanDeps
): Promise<ControlLayerGraphInput[]> => {
  const composites = planControlComposites(document, bbox);
  if (composites.length === 0) {
    return [];
  }

  const layerById = new Map(document.layers.map((layer) => [layer.id, layer]));
  const inputs: ControlLayerGraphInput[] = [];
  let controlLoraCount = 0;
  let zImageControlCount = 0;

  for (const { entry, layerId } of composites) {
    const layer = layerById.get(layerId);
    if (!layer || layer.type !== 'control') {
      continue;
    }

    const { adapter } = layer;
    const resolved = adapter.model ? models?.find((candidate) => candidate.key === adapter.model) : undefined;
    const rejection = getControlValidationReason({
      adapterModel: resolved ? { base: resolved.base, type: resolved.type } : null,
      beginEndStepPct: adapter.beginEndStepPct,
      controlLoraIndex: adapter.kind === 'control_lora' ? controlLoraCount : 0,
      kind: adapter.kind,
      mainBase: model.base,
      mainVariant: model.variant ?? undefined,
      weight: adapter.weight,
      zImageControlIndex: adapter.kind === 'z_image_control' ? zImageControlCount : 0,
    });
    if (rejection || !resolved) {
      throw new Error(`[${rejection ?? 'missing_model'}] Control layer "${layer.name}" is invalid.`);
    }
    if (adapter.kind === 'control_lora') {
      controlLoraCount += 1;
    }
    if (adapter.kind === 'z_image_control') {
      zImageControlCount += 1;
    }

    const result = await executeControlComposite(entry, executorDeps);
    inputs.push({
      beginEndStepPct: adapter.beginEndStepPct,
      controlMode: adapter.controlMode,
      id: layerId,
      imageName: result.imageName,
      kind: adapter.kind,
      model: {
        base: resolved.base,
        key: resolved.key,
        name: resolved.name,
        type: resolved.type,
        ...(typeof resolved.hash === 'string' ? { hash: resolved.hash } : {}),
      },
      weight: adapter.weight,
    });
  }

  return inputs;
};

/** FLUX Redux image-influence → backend redux settings (mirrors `graph.ts` FLUX_REDUX_INFLUENCE). */
const FLUX_REDUX_INFLUENCE_SETTINGS = {
  lowest: { downsampling_factor: 5, weight: 1 },
  low: { downsampling_factor: 4, weight: 1 },
  medium: { downsampling_factor: 3, weight: 1 },
  high: { downsampling_factor: 2, weight: 1 },
  highest: { downsampling_factor: 1, weight: 1 },
} as const;

/** Resolves a regional-guidance layer's reference images into graph inputs (drops incomplete/incompatible ones). */
export const resolveRegionalReferenceImages = (
  region: { referenceImages: GenerateReferenceImage[] },
  base: string
): RegionalReferenceImageInput[] => {
  const inputs: RegionalReferenceImageInput[] = [];
  for (const ref of region.referenceImages) {
    if (!ref.isEnabled) {
      continue;
    }
    const { config } = ref;
    if (config.type === 'ip_adapter' && base !== 'flux') {
      if (!config.image || !config.model || config.model.base !== base) {
        continue;
      }
      inputs.push({
        beginEndStepPct: config.beginEndStepPct,
        clipVisionModel: config.clipVisionModel,
        id: ref.id,
        imageName: config.image.imageName,
        method: config.method,
        model: {
          base: config.model.base,
          key: config.model.key,
          name: config.model.name,
          type: config.model.type,
          ...(typeof config.model.hash === 'string' ? { hash: config.model.hash } : {}),
        },
        type: 'ip_adapter',
        weight: config.weight,
      });
    } else if (config.type === 'flux_redux' && base === 'flux') {
      if (!config.image || !config.model || config.model.base !== base) {
        continue;
      }
      inputs.push({
        id: ref.id,
        imageName: config.image.imageName,
        model: {
          base: config.model.base,
          key: config.model.key,
          name: config.model.name,
          type: config.model.type,
          ...(typeof config.model.hash === 'string' ? { hash: config.model.hash } : {}),
        },
        settings: FLUX_REDUX_INFLUENCE_SETTINGS[config.imageInfluence],
        type: 'flux_redux',
      });
    }
  }
  return inputs;
};

/**
 * Composites + resolves the enabled regional-guidance regions into graph inputs.
 * Each region with mask content is composited SEPARATELY (its alpha feeds its own
 * `alpha_mask_to_tensor`); regions invalid for the base (unsupported base, empty,
 * FLUX + negative/autoNegative) are skipped silently, mirroring legacy. Regions
 * whose prompts + reference images are all empty are also skipped.
 */
const collectRegionalGuidanceInputs = async (
  document: CanvasDocumentContractV2,
  bbox: { x: number; y: number; width: number; height: number },
  model: GenerateModelConfig,
  executorDeps: ExecuteCompositePlanDeps
): Promise<RegionalGuidanceInput[]> => {
  if (!isRegionalGuidanceSupportedForBase(model.base)) {
    return [];
  }
  const composites = planRegionalMaskComposites(document, bbox);
  if (composites.length === 0) {
    return [];
  }

  const layerById = new Map(document.layers.map((layer) => [layer.id, layer]));
  const inputs: RegionalGuidanceInput[] = [];

  for (const { entry, layerId } of composites) {
    const layer = layerById.get(layerId);
    if (!layer || layer.type !== 'regional_guidance') {
      continue;
    }

    const referenceImages = resolveRegionalReferenceImages(layer, model.base);
    const rejection = getRegionalGuidanceRejectionReason({
      autoNegative: layer.autoNegative,
      hasContent: true,
      layerName: layer.name,
      mainBase: model.base,
      negativePrompt: layer.negativePrompt,
      positivePrompt: layer.positivePrompt,
      referenceImageCount: referenceImages.length,
    });
    if (rejection) {
      continue;
    }

    const result = await executeRegionalMaskComposite(entry, executorDeps);
    inputs.push({
      autoNegative: layer.autoNegative,
      id: layerId,
      maskImageName: result.imageName,
      negativePrompt: layer.negativePrompt,
      positivePrompt: layer.positivePrompt,
      referenceImages,
    });
  }

  return inputs;
};

/**
 * Runs the canvas-invoke pipeline with injected dependencies. Never throws:
 * every failure is turned into a notice, and the in-flight guard is always
 * cleared. Returns when the graph has been dispatched (or the invoke aborted).
 */
export const runCanvasInvocation = async (deps: RunCanvasInvocationDeps): Promise<void> => {
  const { dispatch, inFlight, projectId } = deps;

  // Concurrency guard: ignore a re-invoke while a prior prepare for this project
  // is still settling (the hotkey can be mashed). Cleared in `finally`.
  if (inFlight.has(projectId)) {
    return;
  }
  inFlight.add(projectId);

  let rasterSnapshot: Extract<CaptureRasterSnapshotResult, { status: 'ok' }>['snapshot'] | null = null;
  try {
    const values = normalizeGenerateWidgetValues(deps.generateValues);
    if (!values) {
      recordNotice(dispatch, 'error', 'Select a supported model before invoking the canvas.');
      return;
    }

    // Resolve model + settings exactly as the generate path does (sync to loaded
    // models, then resolve the seed) so canvas and generate stay in lockstep.
    const synced = deps.models ? syncGenerateWidgetValuesWithModels(values, deps.models) : values;
    const settings = { ...synced, seed: resolveGenerateSeed(synced) };
    const model: GenerateModelConfig = settings.model;

    // Persist any pending paint bitmaps BEFORE compositing from their pixels.
    await deps.flushPendingUploads();

    // Re-read the document AFTER the flush barrier: the flush dispatches
    // `updateCanvasLayerSource` for each just-persisted paint layer, so the
    // pre-flush snapshot still references `paint:empty`/stale bitmap names. Plan,
    // composite, and compile from the post-flush document so the dedupe keys are
    // built from the persisted refs (silent cache misses otherwise) and so the
    // empty-paint-layer filter (see `planComposites`) sees the real `bitmap`.
    // The bbox can also move during a slow flush; take it from the fresh snapshot.
    const documentSnapshot = deps.captureDocumentSnapshot();
    if (!documentSnapshot) {
      recordNotice(dispatch, 'error', 'The canvas has no active document to generate from.');
      return;
    }
    const postFlushDocument = documentSnapshot.canvas.document;
    const bbox = postFlushDocument.bbox;

    const plan = planComposites(postFlushDocument, bbox);
    const controlPlan = planControlComposites(postFlushDocument, bbox);
    const regionalPlan = planRegionalMaskComposites(postFlushDocument, bbox);

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
    const capture = await deps.captureRasterSnapshot(documentSnapshot, [...requiredLayerIds], { signal: deps.signal });
    if (capture.status !== 'ok') {
      recordNotice(
        dispatch,
        'error',
        capture.status === 'stale'
          ? 'The canvas changed while generation was being prepared. Please invoke again.'
          : `The canvas could not be captured for generation (${capture.status}).`
      );
      return;
    }
    rasterSnapshot = capture.snapshot;
    const canvasSnapshot = rasterSnapshot.canvas;
    const getLayerSurface = (layerId: string) => {
      const detached = rasterSnapshot?.layerSurfaces.get(layerId);
      if (!detached) {
        return Promise.reject(new Error(`Canvas raster snapshot is missing layer ${layerId}.`));
      }
      return Promise.resolve(detached);
    };
    const transactionDedupe: CompositeDedupeCache = {
      byHash: new Map(deps.dedupe.byHash),
      byKey: new Map(deps.dedupe.byKey),
    };
    // Bounds-only pre-pass (pure geometry, no upload): content that does not
    // overlap the bbox is txt2img no matter the coverage. Also naturally skips a
    // zero-area bbox (rectsIntersect is false), which validation rejects anyway.
    const contentBounds = computeCompositeContentBounds(plan);

    const executorDeps = { ...deps.executorDeps, dedupe: transactionDedupe, getLayerSurface };

    let mode: CanvasCompileMode = 'txt2img';
    let compositeImageName: string | null = null;
    let maskImageName: string | null = null;
    let noiseMaskImageName: string | null = null;

    if (contentBounds && rectsIntersect(contentBounds, bbox)) {
      const result = await executeCompositePlan(plan, executorDeps);
      compositeImageName = result.base.imageName;

      // The grayscale denoise-limit mask (when enabled inpaint masks exist) both
      // decides inpaint-vs-img2img (its coverage) and feeds the graph. Legacy
      // parity: raster opaque + mask has content → inpaint; else img2img.
      const maskEntry = plan.entries.find((entry) => entry.kind === 'inpaint-mask');
      const maskResult = maskEntry ? await executeMaskComposite(maskEntry, executorDeps) : null;

      const detected = detectCanvasMode({
        bbox,
        bboxFullyCovered: result.bboxFullyCovered,
        contentBounds: result.contentBounds,
        hasActiveInpaintMask: maskResult?.hasContent ?? false,
      });
      mode = detected;

      if (detected === 'inpaint' || detected === 'outpaint') {
        // A real (non-white) mask feeds create_gradient_mask; outpaint without one
        // derives its mask from the raster alpha (maskImageName stays null).
        maskImageName = maskResult?.hasContent ? maskResult.imageName : null;
        const noiseEntry = plan.entries.find((entry) => entry.kind === 'noise-mask');
        if (noiseEntry) {
          noiseMaskImageName = (await executeMaskComposite(noiseEntry, executorDeps)).imageName;
        }
      }
    }

    // Control layers apply in every mode, independent of the base composite —
    // composite + resolve them regardless of whether raster content overlaps.
    const controlLayers = await collectControlLayerInputs(postFlushDocument, bbox, model, deps.models, executorDeps);

    // Regional guidance also applies in every mode, independent of the base
    // composite — composite + resolve each region regardless of raster overlap.
    const regionalGuidance = await collectRegionalGuidanceInputs(postFlushDocument, bbox, model, executorDeps);

    const compiled = compileCanvasGraph({
      bbox,
      compositeImageName,
      compositing: deps.compositing,
      controlLayers,
      destination: deps.destination,
      maskImageName,
      mode,
      model,
      noiseMaskImageName,
      projectSettings: deps.projectSettings,
      regionalGuidance,
      settings,
      strength: deps.strength,
    });

    dispatch({
      backendSupportsCancellation: true,
      destination: deps.destination,
      generate: {
        negativePromptNodeId: compiled.negativePromptNodeId,
        positivePromptNodeId: compiled.positivePromptNodeId,
        seedNodeId: compiled.seedNodeId,
        values: settings,
      },
      graph: compiled.graph,
      canvas: canvasSnapshot,
      projectId,
      type: 'submitCanvasInvocationSnapshot',
    });
    for (const [key, value] of transactionDedupe.byHash) {
      deps.dedupe.byHash.set(key, value);
    }
    for (const [key, value] of transactionDedupe.byKey) {
      deps.dedupe.byKey.set(key, value);
    }
  } catch (error) {
    recordNotice(dispatch, 'error', error instanceof Error ? error.message : String(error));
  } finally {
    rasterSnapshot?.release();
    inFlight.delete(projectId);
  }
};

// ---- Command-layer wiring (per-engine caches + in-flight guard) -------------

/** The dedupe cache's per-map entry cap (bounds unbounded growth across invokes). */
const DEDUPE_CACHE_CAPACITY = 16;

/**
 * A {@link Map} that evicts its oldest (insertion-order) entry once it exceeds
 * `capacity`, so a long-lived dedupe cache can't grow without bound across many
 * invokes with distinct plans/pixels.
 */
class BoundedMap<K, V> extends Map<K, V> {
  constructor(private readonly capacity: number) {
    super();
  }

  override set(key: K, value: V): this {
    if (!this.has(key) && this.size >= this.capacity) {
      const oldest = this.keys().next().value;
      if (oldest !== undefined) {
        this.delete(oldest);
      }
    }
    return super.set(key, value);
  }
}

/** A size-capped {@link CompositeDedupeCache}. */
export const createBoundedCompositeDedupeCache = (capacity = DEDUPE_CACHE_CAPACITY): CompositeDedupeCache => ({
  byHash: new BoundedMap(capacity),
  byKey: new BoundedMap(capacity),
});

// Project ids with a prepare in flight (survives across the async orchestrator).
const inFlightProjects = new Set<string>();
type CanvasInvocationEngine = {
  readonly document: CanvasDocumentCapability;
  readonly exports: CanvasExportCapability;
  readonly lifecycle: CanvasLifecycleCapability;
};
// Per-engine dedupe caches, dropped when the engine is disposed and GC'd.
const dedupeCachesByEngine = new WeakMap<CanvasInvocationEngine, CompositeDedupeCache>();

const getEngineDedupeCache = (engine: CanvasInvocationEngine): CompositeDedupeCache => {
  let cache = dedupeCachesByEngine.get(engine);
  if (!cache) {
    cache = createBoundedCompositeDedupeCache();
    dedupeCachesByEngine.set(engine, cache);
  }
  return cache;
};

/** Arguments for the thin command-layer {@link prepareCanvasInvocation}. */
export interface PrepareCanvasInvocationArgs {
  projectId: string;
  /** The resolved result destination (Canvas staging vs. a durable Gallery image). */
  destination: ResultDestination;
  generateValues: Record<string, unknown>;
  models?: readonly ModelConfig[];
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>;
  strength: number;
  signal?: AbortSignal;
  /**
   * Persisted compositing settings, already defaulted + clamped by the caller
   * (`readCanvasCompositingSettings`). Falls back to legacy defaults when omitted.
   */
  compositing?: CanvasCompositingSettings;
  dispatch: (action: WorkbenchAction) => void;
}

/**
 * Resolves the active project's engine and runs the canvas-invoke orchestrator.
 * Fire-and-track: the caller does not await it (the Invoke command stays sync).
 */
export const prepareCanvasInvocation = async (args: PrepareCanvasInvocationArgs): Promise<void> => {
  const engine = getEngine(args.projectId);
  if (!engine) {
    recordNotice(args.dispatch, 'error', 'Open the canvas before invoking it.');
    return;
  }

  await runCanvasInvocation({
    compositing: args.compositing ?? DEFAULT_CANVAS_COMPOSITING,
    dedupe: getEngineDedupeCache(engine),
    destination: args.destination,
    dispatch: args.dispatch,
    executorDeps: engine.exports.getCompositeExecutorDeps(),
    captureDocumentSnapshot: () => engine.document.captureSnapshot(),
    captureRasterSnapshot: (snapshot, layerIds, options) =>
      engine.exports.captureRasterSnapshot(snapshot, layerIds, options),
    flushPendingUploads: () => engine.lifecycle.flushPendingUploads(),
    generateValues: args.generateValues,
    inFlight: inFlightProjects,
    models: args.models,
    projectId: args.projectId,
    projectSettings: args.projectSettings,
    signal: args.signal ?? new AbortController().signal,
    strength: args.strength,
  });
};
