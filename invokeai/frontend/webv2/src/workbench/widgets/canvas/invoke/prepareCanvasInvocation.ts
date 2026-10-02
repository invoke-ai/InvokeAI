/**
 * Resolve the active engine, flush paint uploads, compose the fresh snapshot, compile, and enqueue. Invalid
 * controls block; rejected regions skip. Publish dedupe only after dispatch succeeds and report failures as
 * notices. A per-project in-flight guard prevents overlapping preparation.
 */

import type { CanvasScalingSettings, GenerateModelConfig } from '@features/generation/contracts';
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

import { compileCanvasGraph, type CanvasCompileMode } from '@features/generation/canvasGraph';
import {
  detectCanvasMode,
  getControlValidationReason,
  getControlValidationReasonMessage,
  getRegionalGuidanceRejectionReason,
  getRegionalGuidanceSupport,
  type ControlLayerGraphInput,
  type ControlValidationReason,
  type RegionalGuidanceInput,
  type RegionalReferenceImageInput,
  resolveGenerateSeed,
} from '@features/generation/graph';
import { normalizeGenerateWidgetValues, syncGenerateWidgetValuesWithModels } from '@features/generation/settings';
import { getGenerationDevicesSnapshot, resolveRandDeviceMetadata } from '@features/queue/devices';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { getCanvasEngine, getCanvasOperations } from '@workbench/canvas-operations/api';
import {
  DEFAULT_CANVAS_COMPOSITING,
  type CanvasCompositingSettings,
} from '@workbench/widgets/canvas/invoke/canvasCompositing';

import { readCanvasScaling } from './canvasScaling';

/** Title on every canvas-invoke failure notice. */
export const CANVAS_INVOKE_ERROR_TITLE = 'Canvas generation failed';

/** Structured control-layer rejection: callers map `code` to localized text. */
export class ControlLayerValidationError extends Error {
  constructor(
    readonly code: ControlValidationReason,
    readonly layerName: string
  ) {
    super(getControlValidationReasonMessage(code, layerName));
    this.name = 'ControlLayerValidationError';
  }
}

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
  /** Capture and upload generation composites, releasing pixel resources on success or failure. */
  composeForGeneration: (options: ComposeForGenerationOptions) => Promise<ComposeForGenerationResult>;
  /** Expanded positive prompts, resolved by the caller before submitting. */
  positivePrompts?: string[];
  /** Cancels raster capture for this invocation. */
  signal: AbortSignal;
  /** Paint-bitmap persistence barrier, awaited before compositing. */
  flushPendingUploads: () => Promise<void>;
  /** Project ids with a prepare currently in flight (module/registry-scoped). */
  inFlight: Set<string>;
  /** Account-qualified concurrency key; defaults to `projectId` for isolated orchestrator callers. */
  inFlightKey?: string;
  /** The generate widget's raw persisted values (model/prompt/steps, shared with Generate). */
  generateValues: Record<string, unknown>;
  /** Loaded models, for the same value/model sync the generate path performs. */
  models?: readonly ModelConfig[];
  /** Project settings (only `useCpuNoise` is consulted by the compiler). */
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>;
  /** Runtime-derived accelerator family recorded in generation metadata. */
  randDevice?: string;
  /** Persisted denoising strength (already defaulted + clamped). Used for every image mode. */
  strength: number;
  /** Persisted compositing settings (infill / coherence / mask blur / output policy), defaulted + clamped. */
  compositing: CanvasCompositingSettings;
  /** Persisted "scale before processing" policy, defaulted. */
  scaling: CanvasScalingSettings;
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  /** Localizes a control-layer rejection; defaults to the English validation sentence. */
  formatControlLayerError?: (code: ControlValidationReason, layerName: string) => string;
}

const recordNotice = (
  notifications: WorkbenchNotificationCommands,
  kind: WorkbenchNotificationKind,
  message: string
): void => {
  notifications.add({ kind, message, title: CANVAS_INVOKE_ERROR_TITLE });
};

/**
 * Validate enabled content-bearing controls in z-order and resolve adapters; invalid controls throw and block
 * invocation. Join uploaded names with captured model metadata for graph inputs.
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
        throw new ControlLayerValidationError(rejection ?? 'missing_model', layer.name);
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
  const kind = getRegionalGuidanceSupport(base)?.referenceImages ?? null;
  for (const ref of region.referenceImages) {
    if (!ref.isEnabled) {
      continue;
    }
    const { config } = ref;
    if (config.type === 'ip_adapter' && kind === 'ip_adapter') {
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
    } else if (config.type === 'flux_redux' && kind === 'flux_redux') {
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
 * Resolve regional references and silently skip unsupported/noncontributing regions. Join uploaded masks with
 * captured prompt/reference metadata for graph inputs.
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
      const referenceImages = resolveRegionalReferenceImages(layer, model.base);
      const rejection = getRegionalGuidanceRejectionReason({
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
  const inFlightKey = deps.inFlightKey ?? projectId;

  // Ignore overlapping preparation for the same project; finally clears the guard.
  if (inFlight.has(inFlightKey)) {
    return;
  }
  inFlight.add(inFlightKey);

  try {
    const values = normalizeGenerateWidgetValues(deps.generateValues);
    if (!values) {
      recordNotice(commands.notifications, 'error', 'Select a supported model before invoking the canvas.');
      return;
    }

    // Sync loaded models and resolve seeds identically to the Generate path.
    const synced = deps.models ? syncGenerateWidgetValuesWithModels(values, deps.models) : values;
    const settings = { ...synced, seed: resolveGenerateSeed(synced) };
    const model: GenerateModelConfig = settings.model;

    // Flush persisted paint before capture so planning sees current bitmap references and bbox.
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
    deps.signal.throwIfAborted();
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
      randDevice: deps.randDevice,
      regionalGuidance: regions.toGraphInputs(composites.regionalMaskImages),
      scaling: deps.scaling,
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
      positivePrompts: deps.positivePrompts,
      projectId,
    });
    composed.dedupeCommit.commit();
  } catch (error) {
    if (deps.signal.aborted) {
      return;
    }

    const message =
      error instanceof ControlLayerValidationError && deps.formatControlLayerError
        ? deps.formatControlLayerError(error.code, error.layerName)
        : error instanceof Error
          ? error.message
          : String(error);
    recordNotice(commands.notifications, 'error', message);
  } finally {
    inFlight.delete(inFlightKey);
  }
};

// Project ids with a prepare in flight (survives across the async orchestrator).
const inFlightProjects = new Set<string>();

registerAccountOwnedResource({
  clear: () => {
    inFlightProjects.clear();
  },
  name: 'canvas-invocation-guards',
});

/** Arguments for the thin command-layer {@link prepareCanvasInvocation}. */
export interface PrepareCanvasInvocationArgs {
  projectId: string;
  /** The resolved result destination (Canvas staging vs. a durable Gallery image). */
  destination: ResultDestination;
  generateValues: Record<string, unknown>;
  models?: readonly ModelConfig[];
  /** Caller-captured identity lifetime; direct synchronous callers may omit it. */
  owner?: AccountScope;
  /** Expanded positive prompts, resolved by the caller before submitting. */
  positivePrompts?: string[];
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>;
  /**
   * The canvas widget's persisted values. The scaling policy is read here, in
   * the lazily loaded orchestrator, so its reader stays out of the eager submit path.
   */
  canvasValues?: Record<string, unknown>;
  strength: number;
  signal?: AbortSignal;
  /**
   * Persisted compositing settings, already defaulted + clamped by the caller
   * (`readCanvasCompositingSettings`). Falls back to legacy defaults when omitted.
   */
  compositing?: CanvasCompositingSettings;
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  /** Localizes a control-layer rejection; defaults to the English validation sentence. */
  formatControlLayerError?: (code: ControlValidationReason, layerName: string) => string;
}

/**
 * Settle after dispatch or reported failure so the active command keeps its preparing acknowledgement through the
 * entire pre-queue window.
 */
export const prepareCanvasInvocation = async (args: PrepareCanvasInvocationArgs): Promise<void> => {
  const owner = args.owner ?? captureAccountScope();
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const engine = getCanvasEngine(args.projectId);
  if (!engine) {
    recordNotice(args.commands.notifications, 'error', 'Open the canvas before invoking it.');
    return;
  }
  const operations = getCanvasOperations(engine);
  const signal = args.signal ? AbortSignal.any([args.signal, owner.signal]) : owner.signal;

  await runCanvasInvocation({
    compositing: args.compositing ?? DEFAULT_CANVAS_COMPOSITING,
    destination: args.destination,
    commands: args.commands,
    composeForGeneration: (composeOptions) => operations.composeForGeneration(composeOptions),
    flushPendingUploads: () => engine.lifecycle.flushPendingUploads(),
    formatControlLayerError: args.formatControlLayerError,
    generateValues: args.generateValues,
    inFlight: inFlightProjects,
    inFlightKey: `${owner.epoch}:${args.projectId}`,
    models: args.models,
    projectId: args.projectId,
    projectSettings: args.projectSettings,
    randDevice: resolveRandDeviceMetadata(args.projectSettings.useCpuNoise, getGenerationDevicesSnapshot().options),
    signal,
    scaling: readCanvasScaling(args.canvasValues),
    strength: args.strength,
  });
};
