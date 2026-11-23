/** Flush, compose, compile and enqueue; dedupe publishes only after dispatch, one run per project at a time. */

import type {
  CanvasScalingSettings,
  GenerateModelConfig,
  GenerationProjectSettings,
} from '@features/generation/contracts';
import type { ControlLayerGraphInput, RegionalGuidanceInput } from '@features/generation/graph';
import type { ModelConfig } from '@features/models';
import type { ComposeForGenerationOptions, ComposeForGenerationResult } from '@workbench/canvas-operations/api';
import type { CanvasInputIssue, ControlLayerVerdict, RegionalLayerVerdict } from '@workbench/canvasGenerationInputs';
import type { ControlModelMemory } from '@workbench/controlModelResolution';
import type { ExpandedPositivePrompts, ResultDestination } from '@workbench/invocationContracts';
import type { CanvasCompositingSettings } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import type { WorkbenchCommands, WorkbenchNotificationCommands } from '@workbench/workbenchStore';

import { compileCanvasGraph, type CanvasCompileMode } from '@features/generation/canvasGraph';
import { detectCanvasMode, resolveGenerateSeed } from '@features/generation/graph';
import { normalizeGenerateWidgetValues, syncGenerateWidgetValuesWithModels } from '@features/generation/settings';
import { getGenerationDevicesSnapshot, resolveRandDeviceMetadata } from '@features/queue/devices';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { getCanvasEngine, getCanvasOperations, recordCanvasMetadata } from '@workbench/canvas-operations/api';
import { createCanvasInputJudge } from '@workbench/canvasGenerationInputs';
import { getWorkbenchPreferences } from '@workbench/settings/store';

import type { CanvasInvocationFailure, CanvasInvocationNotice } from './canvasInvocationFailure';

import { readCanvasScaling } from './canvasScaling';

/** Composition aborts and rethrows this, carrying the blocking layer out. */
class CanvasInputRefusal extends Error {
  constructor(readonly issue: CanvasInputIssue) {
    super(`Canvas input refused: ${issue.kind}.`);
    this.name = 'CanvasInputRefusal';
  }
}

/** Injected dependencies for the React-free orchestrator. */
export interface RunCanvasInvocationDeps {
  /** The active project id (also the in-flight guard key). */
  projectId: string;
  /** Drives the output node's `is_intermediate` and the snapshot's destination, so Canvas can target Gallery. */
  destination: ResultDestination;
  /** Capture and upload generation composites, releasing pixel resources on success or failure. */
  composeForGeneration: (options: ComposeForGenerationOptions) => Promise<ComposeForGenerationResult>;
  /** Expanded by the caller before submitting. */
  expansion?: ExpandedPositivePrompts;
  /** Invoke Front: the backend runs the batch ahead of the user's other pending work. */
  prepend?: boolean;
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
  /** Remembered control-model picks, which resolve control layers left on automatic. */
  controlModelMemory: ControlModelMemory;
  projectSettings: GenerationProjectSettings;
  /** Runtime-derived accelerator family recorded in generation metadata. */
  randDevice?: string;
  /** Persisted denoising strength (already defaulted + clamped). Used for every image mode. */
  strength: number;
  /** Persisted compositing settings (infill / coherence / mask blur / output policy), defaulted + clamped. */
  compositing: CanvasCompositingSettings;
  /** Persisted "scale before processing" policy, defaulted. */
  scaling: CanvasScalingSettings;
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  formatCanvasInvocationFailure: (failure: CanvasInvocationFailure) => CanvasInvocationNotice;
}

const recordFailure = (
  notifications: WorkbenchNotificationCommands,
  formatCanvasInvocationFailure: RunCanvasInvocationDeps['formatCanvasInvocationFailure'],
  failure: CanvasInvocationFailure
): void => {
  notifications.add({ kind: 'error', ...formatCanvasInvocationFailure(failure) });
};

const admitLayer = <Verdict extends ControlLayerVerdict | RegionalLayerVerdict>(
  verdict: Verdict,
  layerId: string,
  inputs: Map<string, Extract<Verdict, { status: 'contributes' }>['input']>
): boolean => {
  if (verdict.status === 'contributes') {
    inputs.set(layerId, verdict.input);
    return true;
  }
  if (verdict.status === 'skipped') {
    return false;
  }
  throw new CanvasInputRefusal(verdict.issue);
};

/** Never throws: failures become notices and the in-flight guard always clears. */
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
      recordFailure(commands.notifications, deps.formatCanvasInvocationFailure, { kind: 'no_model' });
      return;
    }

    // Sync loaded models and resolve seeds identically to the Generate path.
    const synced = deps.models ? syncGenerateWidgetValuesWithModels(values, deps.models) : values;
    const settings = { ...synced, seed: resolveGenerateSeed(synced) };
    const model: GenerateModelConfig = settings.model;

    // Flush persisted paint before capture so planning sees current bitmap references and bbox.
    await deps.flushPendingUploads();

    const judge = createCanvasInputJudge({
      cfgScale: settings.cfgScale,
      controlModelMemory: deps.controlModelMemory,
      model,
      models: deps.models,
    });
    const controlInputs = new Map<string, Omit<ControlLayerGraphInput, 'imageName'>>();
    const regionalInputs = new Map<string, Omit<RegionalGuidanceInput, 'maskImageName' | 'maskBounds'>>();
    const composed = await deps.composeForGeneration({
      // Mode-union drift guard: `detectCanvasMode`'s facts/mode types must stay structurally identical to
      // canvas-operations' `GenerationModeFacts` / `GenerationCompositeMode` — tsc fails here on drift in either
      // direction. A hosted model takes the frame's pixels as they are, with no masked tail: bounds that overlap the
      // frame with nothing visible inside it leave an empty frame, which it generates from text.
      detectMode: (facts) =>
        model.type === 'external_image_generator' && !facts.bboxHasContent ? 'txt2img' : detectCanvasMode(facts),
      shouldCompositeControlLayer: (layer) => admitLayer(judge.control(layer), layer.id, controlInputs),
      shouldCompositeRegionalMask: (layer) => admitLayer(judge.regional(layer), layer.id, regionalInputs),
      signal: deps.signal,
    });
    deps.signal.throwIfAborted();
    if (composed.status !== 'ok') {
      recordFailure(commands.notifications, deps.formatCanvasInvocationFailure, { kind: 'capture', result: composed });
      return;
    }
    const { composites } = composed;
    // The other half of the mode-union drift guard (see `detectMode` above).
    const mode: CanvasCompileMode = composites.mode;

    const controlLayers = composites.controlImages.flatMap(({ imageName, layerId }) => {
      const input = controlInputs.get(layerId);
      return input ? [{ ...input, imageName }] : [];
    });
    const compiled = compileCanvasGraph({
      canvasMetadata: recordCanvasMetadata(composites.canvas.document, {
        controls: controlLayers,
        models: deps.models,
        settings: { strength: deps.strength, compositing: deps.compositing, scaling: deps.scaling },
      }),
      bbox: composites.bbox,
      compositeImageName: composites.baseImageName,
      compositing: deps.compositing,
      controlLayers,
      destination: deps.destination,
      maskImageName: composites.maskImageName,
      mode,
      model,
      noiseMaskImageName: composites.noiseMaskImageName,
      projectSettings: deps.projectSettings,
      randDevice: deps.randDevice,
      regionalGuidance: composites.regionalMaskImages.flatMap(({ bounds, imageName, layerId }) => {
        const input = regionalInputs.get(layerId);
        return input ? [{ ...input, maskBounds: bounds, maskImageName: imageName }] : [];
      }),
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
      ...deps.expansion,
      ...(deps.prepend ? { prepend: true } : {}),
      projectId,
    });
    composed.dedupeCommit.commit();
  } catch (error) {
    if (deps.signal.aborted) {
      return;
    }

    recordFailure(
      commands.notifications,
      deps.formatCanvasInvocationFailure,
      error instanceof CanvasInputRefusal
        ? { issue: error.issue, kind: 'canvas_input' }
        : { kind: 'error', message: error instanceof Error ? error.message : String(error) }
    );
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

export interface PrepareCanvasInvocationArgs {
  projectId: string;
  /** The resolved result destination (Canvas staging vs. a durable Gallery image). */
  destination: ResultDestination;
  generateValues: Record<string, unknown>;
  models?: readonly ModelConfig[];
  /** Caller-captured identity lifetime; direct synchronous callers may omit it. */
  owner?: AccountScope;
  expansion?: ExpandedPositivePrompts;
  prepend?: boolean;
  projectSettings: GenerationProjectSettings;
  /** Read here, in the lazy orchestrator, so the scaling reader stays out of the eager submit path. */
  canvasValues?: Record<string, unknown>;
  strength: number;
  signal?: AbortSignal;
  /** From resolved canvas values, so user defaults apply. */
  compositing: CanvasCompositingSettings;
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  formatCanvasInvocationFailure: RunCanvasInvocationDeps['formatCanvasInvocationFailure'];
}

/** Settles after dispatch or a reported failure, keeping the preparing acknowledgement until queued. */
export const prepareCanvasInvocation = async (args: PrepareCanvasInvocationArgs): Promise<void> => {
  const owner = args.owner ?? captureAccountScope();
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const engine = getCanvasEngine(args.projectId);
  if (!engine) {
    recordFailure(args.commands.notifications, args.formatCanvasInvocationFailure, { kind: 'canvas_not_open' });
    return;
  }
  const operations = getCanvasOperations(engine);
  const signal = args.signal ? AbortSignal.any([args.signal, owner.signal]) : owner.signal;

  await runCanvasInvocation({
    compositing: args.compositing,
    destination: args.destination,
    commands: args.commands,
    expansion: args.expansion,
    composeForGeneration: (composeOptions) => operations.composeForGeneration(composeOptions),
    controlModelMemory: getWorkbenchPreferences().controlModelMemory,
    flushPendingUploads: () => engine.lifecycle.flushPendingUploads(),
    formatCanvasInvocationFailure: args.formatCanvasInvocationFailure,
    generateValues: args.generateValues,
    inFlight: inFlightProjects,
    inFlightKey: `${owner.epoch}:${args.projectId}`,
    models: args.models,
    ...(args.prepend ? { prepend: true } : {}),
    projectId: args.projectId,
    projectSettings: args.projectSettings,
    randDevice: resolveRandDeviceMetadata(args.projectSettings.useCpuNoise, getGenerationDevicesSnapshot().options),
    signal,
    scaling: readCanvasScaling(args.canvasValues),
    strength: args.strength,
  });
};
