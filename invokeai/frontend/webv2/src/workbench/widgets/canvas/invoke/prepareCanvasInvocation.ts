/**
 * Wires the canvas-generation pipeline into the Invoke button.
 *
 * `prepareCanvasInvocation` is the thin command-layer entry: it resolves the
 * active project's engine and delegates to the React-free
 * {@link runCanvasInvocation} orchestrator. The orchestrator runs the whole
 * flush → compose → compile → enqueue flow with every side-effecting dependency
 * injected, so it is fully node-testable with fakes.
 *
 * ## Flow (per the plan)
 * 1. Resolve the engine (no engine → notice, abort) and normalize/sync the
 *    generate values exactly as the generate path does.
 * 2. `flushPendingUploads()` — the paint-bitmap persistence barrier — so the
 *    composite reads the latest painted pixels. The composite operation
 *    captures its document snapshot internally, AFTER this barrier, so it
 *    plans from the post-flush document (fresh bitmap refs, fresh bbox).
 * 3. `composeForGeneration` — Canvas's one deep composite call — produces
 *    every uploaded image (base / masks / controls / regionals) plus the
 *    resolved mode. This module supplies the policy: `detectCanvasMode` as the
 *    mode strategy, control validation as a throwing predicate (an invalid
 *    enabled nonempty control layer blocks the invoke), and regional-guidance
 *    rejection as a silent-skip predicate.
 * 4. Compile the base-appropriate graph and dispatch it to canvas staging.
 * 5. Publish the operation-local dedupe cache only after dispatch returns.
 * 6. Any failure (validation throw, upload error, unsupported mode) records a
 *    notice — the app never gets stuck.
 *
 * A module-scoped in-flight guard drops an invoke while a prior prepare for the
 * same project is still running (the Invoke hotkey can be mashed).
 */

import type { GenerateModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  CanvasControlLayerContract,
  CanvasRegionalGuidanceLayerContract,
  RegionalGuidanceReferenceImage,
} from '@workbench/canvas-engine/api';
import type { ComposeForGenerationOptions, ComposeForGenerationResult } from '@workbench/canvas-operations/api';
import type { ResultDestination } from '@workbench/invocationContracts';
import type { WorkbenchNotificationKind } from '@workbench/projectContracts';
import type { ProjectSettings } from '@workbench/settings/contracts';
import type { WorkbenchCommands, WorkbenchNotificationCommands } from '@workbench/workbenchStore';

import {
  compileCanvasGraph,
  detectCanvasMode,
  getControlValidationReason,
  getRegionalGuidanceRejectionReason,
  isRegionalGuidanceSupportedForBase,
  type CanvasCompileMode,
  type ControlLayerGraphInput,
  type RegionalGuidanceInput,
  type RegionalReferenceImageInput,
  resolveGenerateSeed,
} from '@features/generation/graph';
import { normalizeGenerateWidgetValues, syncGenerateWidgetValuesWithModels } from '@features/generation/settings';
import { getCanvasEngine, getCanvasOperations } from '@workbench/canvas-operations/api';
import {
  DEFAULT_CANVAS_COMPOSITING,
  type CanvasCompositingSettings,
} from '@workbench/widgets/canvas/invoke/canvasCompositing';

/** Title on every canvas-invoke failure notice. */
export const CANVAS_INVOKE_ERROR_TITLE = 'Canvas generation failed';

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
  /**
   * Canvas's composite-generation operation: snapshot → plan → capture →
   * composite → upload, releasing pixel resources on both success and failure.
   */
  composeForGeneration: (options: ComposeForGenerationOptions) => Promise<ComposeForGenerationResult>;
  /** Cancels raster capture for this invocation. */
  signal: AbortSignal;
  /** Paint-bitmap persistence barrier, awaited before compositing. */
  flushPendingUploads: () => Promise<void>;
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
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
}

const recordNotice = (
  notifications: WorkbenchNotificationCommands,
  kind: WorkbenchNotificationKind,
  message: string
): void => {
  notifications.add({ kind, message, title: CANVAS_INVOKE_ERROR_TITLE });
};

/**
 * Control-layer policy + metadata side-channel for the composite operation.
 * `shouldComposite` validates each enabled content-bearing control layer in
 * z-order and resolves its adapter model from the loaded models: an invalid
 * layer throws a shared reason code and blocks the invocation (content-less
 * control layers are already excluded by the operation's plan and remain
 * harmless). `toGraphInputs` joins the composited image names back with the
 * recorded adapter/model metadata.
 */
const createControlLayerCollector = (
  model: GenerateModelConfig,
  models: readonly ModelConfig[] | undefined
): {
  shouldComposite(layer: CanvasControlLayerContract): boolean;
  toGraphInputs(images: readonly { layerId: string; imageName: string }[]): ControlLayerGraphInput[];
} => {
  const metadata = new Map<string, Omit<ControlLayerGraphInput, 'imageName'>>();
  let controlLoraCount = 0;
  let zImageControlCount = 0;

  return {
    shouldComposite: (layer) => {
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
      metadata.set(layer.id, {
        beginEndStepPct: adapter.beginEndStepPct,
        controlMode: adapter.controlMode,
        id: layer.id,
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
      return true;
    },
    toGraphInputs: (images) =>
      images.flatMap(({ imageName, layerId }) => {
        const meta = metadata.get(layerId);
        return meta ? [{ ...meta, imageName }] : [];
      }),
  };
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
  region: { referenceImages: RegionalGuidanceReferenceImage[] },
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
 * Regional-guidance policy + metadata side-channel for the composite operation.
 * `shouldComposite` resolves each region's reference images and rejects regions
 * invalid for the base (unsupported base, FLUX + negative/autoNegative, all
 * prompts + references empty) with a SILENT skip, mirroring legacy.
 * `toGraphInputs` joins the composited mask image names back with the recorded
 * prompt/reference metadata.
 */
const createRegionalGuidanceCollector = (
  model: GenerateModelConfig
): {
  shouldComposite(layer: CanvasRegionalGuidanceLayerContract): boolean;
  toGraphInputs(images: readonly { layerId: string; imageName: string }[]): RegionalGuidanceInput[];
} => {
  const metadata = new Map<string, Omit<RegionalGuidanceInput, 'maskImageName'>>();

  return {
    shouldComposite: (layer) => {
      if (!isRegionalGuidanceSupportedForBase(model.base)) {
        return false;
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
        return false;
      }
      metadata.set(layer.id, {
        autoNegative: layer.autoNegative,
        id: layer.id,
        negativePrompt: layer.negativePrompt,
        positivePrompt: layer.positivePrompt,
        referenceImages,
      });
      return true;
    },
    toGraphInputs: (images) =>
      images.flatMap(({ imageName, layerId }) => {
        const meta = metadata.get(layerId);
        return meta ? [{ ...meta, maskImageName: imageName }] : [];
      }),
  };
};

/**
 * Runs the canvas-invoke pipeline with injected dependencies. Never throws:
 * every failure is turned into a notice, and the in-flight guard is always
 * cleared. Returns when the graph has been dispatched (or the invoke aborted).
 */
export const runCanvasInvocation = async (deps: RunCanvasInvocationDeps): Promise<void> => {
  const { commands, inFlight, projectId } = deps;

  // Concurrency guard: ignore a re-invoke while a prior prepare for this project
  // is still settling (the hotkey can be mashed). Cleared in `finally`.
  if (inFlight.has(projectId)) {
    return;
  }
  inFlight.add(projectId);

  try {
    const values = normalizeGenerateWidgetValues(deps.generateValues);
    if (!values) {
      recordNotice(commands.notifications, 'error', 'Select a supported model before invoking the canvas.');
      return;
    }

    // Resolve model + settings exactly as the generate path does (sync to loaded
    // models, then resolve the seed) so canvas and generate stay in lockstep.
    const synced = deps.models ? syncGenerateWidgetValuesWithModels(values, deps.models) : values;
    const settings = { ...synced, seed: resolveGenerateSeed(synced) };
    const model: GenerateModelConfig = settings.model;

    // Persist any pending paint bitmaps BEFORE compositing from their pixels.
    // The composite operation captures its document snapshot internally — after
    // this barrier — so it plans from the post-flush document: the flush
    // dispatches `updateCanvasLayerSource` for each just-persisted paint layer,
    // and a pre-flush snapshot would still reference `paint:empty`/stale bitmap
    // names (silent dedupe cache misses) and a possibly-moved bbox.
    await deps.flushPendingUploads();

    const controls = createControlLayerCollector(model, deps.models);
    const regions = createRegionalGuidanceCollector(model);
    const composed = await deps.composeForGeneration({
      // Mode-union drift guard: `detectCanvasMode`'s facts/mode types must stay
      // structurally identical to canvas-operations' `GenerationModeFacts` /
      // `GenerationCompositeMode` — tsc fails here on drift in either direction.
      detectMode: (facts) => detectCanvasMode(facts),
      shouldCompositeControlLayer: controls.shouldComposite,
      shouldCompositeRegionalMask: regions.shouldComposite,
      signal: deps.signal,
    });
    if (composed.status !== 'ok') {
      recordNotice(
        commands.notifications,
        'error',
        composed.status === 'no-document'
          ? 'The canvas has no active document to generate from.'
          : composed.status === 'stale'
            ? 'The canvas changed while generation was being prepared. Please invoke again.'
            : `The canvas could not be captured for generation (${composed.status}).`
      );
      return;
    }
    const { composites } = composed;
    // The other half of the mode-union drift guard (see `detectMode` above).
    const mode: CanvasCompileMode = composites.mode;

    const compiled = compileCanvasGraph({
      bbox: composites.bbox,
      compositeImageName: composites.baseImageName,
      compositing: deps.compositing,
      controlLayers: controls.toGraphInputs(composites.controlImages),
      destination: deps.destination,
      maskImageName: composites.maskImageName,
      mode,
      model,
      noiseMaskImageName: composites.noiseMaskImageName,
      projectSettings: deps.projectSettings,
      regionalGuidance: regions.toGraphInputs(composites.regionalMaskImages),
      settings,
      strength: deps.strength,
    });

    commands.generation.submitCanvas({
      backendSupportsCancellation: true,
      destination: deps.destination,
      generate: {
        negativePromptNodeId: compiled.negativePromptNodeId,
        positivePromptNodeId: compiled.positivePromptNodeId,
        seedNodeId: compiled.seedNodeId,
        values: settings,
      },
      graph: compiled.graph,
      canvas: composites.canvas,
      projectId,
    });
    composed.dedupeCommit.commit();
  } catch (error) {
    recordNotice(commands.notifications, 'error', error instanceof Error ? error.message : String(error));
  } finally {
    inFlight.delete(projectId);
  }
};

// Project ids with a prepare in flight (survives across the async orchestrator).
const inFlightProjects = new Set<string>();

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
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
}

/**
 * Resolves the active project's engine and runs the canvas-invoke orchestrator.
 * Fire-and-track: the caller does not await it (the Invoke command stays sync).
 */
export const prepareCanvasInvocation = async (args: PrepareCanvasInvocationArgs): Promise<void> => {
  const engine = getCanvasEngine(args.projectId);
  if (!engine) {
    recordNotice(args.commands.notifications, 'error', 'Open the canvas before invoking it.');
    return;
  }
  const operations = getCanvasOperations(engine);

  await runCanvasInvocation({
    compositing: args.compositing ?? DEFAULT_CANVAS_COMPOSITING,
    destination: args.destination,
    commands: args.commands,
    composeForGeneration: (composeOptions) => operations.composeForGeneration(composeOptions),
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
