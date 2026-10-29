/**
 * Offline export cases compile to the graph Generate or Canvas would submit: the model's defaults, the case's values
 * and component picks, required components filled from the installed models the way a user fills them, and the same
 * reconciliation and compilers the widget runs. Callers own the wire format and the Workbench canvas defaults.
 */

import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/core/contracts';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';

import type { ArchitectureCapabilitiesRow } from './architectureCapabilities';
import type { ComponentPolicyContext, ComponentSlotPolicy } from './baseGenerationPolicies';
import type { ControlLayerGraphInput } from './canvas/addControlLayers';
import type { CanvasCompositingSettings, CanvasScalingSettings } from './canvas/types';
import type { BackendGraphContract, GenerationProjectSettings, ResultDestination } from './contracts';
import type { ComponentModelConfig, GenerateModelConfig, GenerateWidgetValues } from './types';

import { resetArchitectureCapabilities, setArchitectureCapabilities } from './architectureCapabilities';
import {
  coerceSchedulerForGraph,
  GENERATE_COMPONENT_VALUE_KEYS,
  getComponentPolicyContext,
  getComponentSectionPolicy,
  getDefaultGenerateSettings,
  getGenerateModelSelectionResult,
  getMissingReferenceImageFormat,
  getSchedulerOptions,
  isArchitectureDescribed,
  isGenerateModelSelectable,
  isLoraSupported,
  isReferenceImageSupported,
  withAddedReferenceImages,
} from './baseGenerationPolicies';
import { compileCanvasGraph } from './canvas/compileCanvasGraph';
import {
  CONTROL_ADAPTER_KINDS,
  isControlKindSupportedForBase,
  isControlModelUsableForKind,
} from './canvas/controlValidation';
import { hasDynamicPromptSyntax, planGenerateSubmission, type GenerateSubmissionPrompts } from './dynamicPrompts';
import { compileGenerateGraph } from './graph';
import { stabilizeBackendGraphNodeIds } from './previewGraph';
import { resolveGenerateWidgetValues } from './resolveGenerateWidgetValues';
import {
  deriveAspectRatioId,
  GENERATE_UI_STATE_KEYS,
  getModelDefaultVae,
  hasModelDefaultVae,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
} from './settings';

/** `txt2img` compiles like Generate; `img2img` and `controlnet` compile like a full-frame Canvas invoke. */
export type GraphExportMode = 'txt2img' | 'img2img' | 'controlnet';

export interface GraphExportCase {
  /** Becomes the graph id. */
  id: string;
  mode: GraphExportMode;
  modelKey: string;
  /** Picks keyed by component slot; required slots left out are filled from the installed models. */
  components: Readonly<Record<string, string>>;
  width: number;
  height: number;
  steps: number;
  seed: number;
  positivePrompt: string;
  negativePrompt: string;
  loras: readonly { key: string; weight: number }[];
  /** Uploaded image names. */
  referenceImages: readonly string[];
  /** An uploaded image already sized width × height, as the Canvas composite always is. */
  initImage: { imageName: string; strength: number } | null;
  /** Each image is an uploaded width × height control composite. */
  control: readonly { modelKey: string; imageName: string; weight: number; begin: number; end: number }[];
  /** Generate settings overrides. Keys the case sets through its own fields are rejected. */
  settings: Readonly<Record<string, unknown>>;
}

export interface GraphExportEnvironment {
  models: readonly ModelConfig[];
  destination: ResultDestination;
  projectSettings: GenerationProjectSettings;
  randDevice: string;
  /** What the Canvas widget supplies from its persisted values. */
  canvas: { compositing: CanvasCompositingSettings; scaling: CanvasScalingSettings; strength: number };
}

export interface CompiledGraphExportCase {
  /** Node ids are stabilized: identical cases compile to identical graphs. */
  graph: BackendGraphContract;
  /** The key of every component model the case compiled with, by settings key. */
  components: Record<string, string>;
  /** What Generate queues for the graph, with the prompt and seed node ids in `graph`. */
  submission: GenerateSubmissionPrompts & {
    negativePromptNodeId: string;
    positivePromptNodeId: string;
    seedNodeId: string;
  };
}

/** Keys a case sets through its own fields, picks as components, or that need backend catalogs or expansion. */
const NON_OVERRIDABLE_SETTINGS: ReadonlySet<string> = new Set([
  'aspectRatioId',
  'aspectRatioValue',
  'dynamicPromptsCombinatorial',
  'dynamicPromptsMaxPrompts',
  'dynamicPromptsResample',
  'dynamicPromptsSampleSeed',
  'dynamicPromptsSeedBehaviour',
  'height',
  'loras',
  'model',
  'modelKey',
  'negativePrompt',
  'positivePrompt',
  'promptTemplate',
  'referenceImages',
  'seed',
  'seedMode',
  'steps',
  'width',
  ...GENERATE_COMPONENT_VALUE_KEYS,
  ...Object.keys(GENERATE_UI_STATE_KEYS),
]);

const listKeys = (keys: readonly string[]): string => (keys.length > 0 ? keys.join(', ') : 'none');

const listModelKeys = (models: readonly ModelConfig[]): string => listKeys(models.map((model) => model.key));

/** Hold `rows` as the capability table, which every Generation policy reads, while `run` compiles. */
export const withArchitectureCapabilities = <T>(rows: readonly ArchitectureCapabilitiesRow[], run: () => T): T => {
  setArchitectureCapabilities(rows);

  try {
    return run();
  } finally {
    resetArchitectureCapabilities();
  }
};

const getSelectableModel = (key: string, models: readonly ModelConfig[]): GenerateModelConfig => {
  const selectable = models.filter(isGenerateModelSelectable).filter(isArchitectureDescribed);
  const model = selectable.find((candidate) => candidate.key === key);

  if (!model) {
    throw new Error(`model_key "${key}" is not a selectable main model. Choices: ${listModelKeys(selectable)}.`);
  }

  return model;
};

/** Selecting the model, then "Reset to model defaults", which also applies the model's default VAE. */
const getModelDefaultValues = (model: GenerateModelConfig, models: readonly ModelConfig[]): GenerateWidgetValues => {
  const { settings } = getGenerateModelSelectionResult({
    currentValues: getDefaultGenerateSettings(model),
    model,
    models,
  });
  const vae = hasModelDefaultVae(model) ? getModelDefaultVae(model, models.filter(isVaeModelConfig)) : settings.vae;

  return { ...settings, model, vae };
};

const withCaseValues = (values: GenerateWidgetValues, input: GraphExportCase): GenerateWidgetValues => ({
  ...values,
  aspectRatioId: deriveAspectRatioId(input.width, input.height),
  aspectRatioValue: input.width / input.height,
  height: input.height,
  negativePrompt: input.negativePrompt,
  positivePrompt: input.positivePrompt,
  seed: input.seed,
  seedMode: 'fixed',
  steps: input.steps,
  width: input.width,
});

const withSettingsOverrides = (
  values: GenerateWidgetValues,
  overrides: GraphExportCase['settings']
): GenerateWidgetValues => {
  const validKeys = Object.keys(values)
    .filter((key) => !NON_OVERRIDABLE_SETTINGS.has(key))
    .sort();
  const unknownKey = Object.keys(overrides).find((key) => !validKeys.includes(key));

  if (unknownKey !== undefined) {
    throw new Error(`Unknown settings key "${unknownKey}". Valid keys: ${listKeys(validKeys)}.`);
  }

  return { ...values, ...overrides };
};

const fitsSlot = (slot: ComponentSlotPolicy, candidate: ModelConfig, ctx: ComponentPolicyContext): boolean =>
  (slot.modelTypes as readonly string[]).includes(candidate.type) && (!slot.filter || slot.filter(candidate, ctx));

/** The value the component picker commits for `model`, by the slot's value kind. */
const toSlotValue = (slot: ComponentSlotPolicy, model: ModelConfig): ComponentModelConfig | null => {
  switch (slot.valueKind) {
    case 'main':
      return isMainModelConfig(model) ? model : null;
    case 'vae':
      return isVaeModelConfig(model) ? model : null;
    case 'component':
      return isModelIdentifierConfig(model) ? model : null;
  }
};

const withComponentPicks = (
  values: GenerateWidgetValues,
  picks: GraphExportCase['components'],
  models: readonly ModelConfig[]
): GenerateWidgetValues => {
  const { slots } = getComponentSectionPolicy(values.model, values);
  let next = values;

  for (const [slotKey, modelKey] of Object.entries(picks)) {
    const slot = slots.find((candidate) => candidate.key === slotKey);

    if (!slot) {
      const validSlots = listKeys(slots.map((candidate) => candidate.key));

      throw new Error(`Unknown component slot "${slotKey}" for ${values.model.name}. Valid slots: ${validSlots}.`);
    }

    const ctx = getComponentPolicyContext(next.model, next);
    const fitting = models.filter((candidate) => fitsSlot(slot, candidate, ctx));
    const pick = fitting.find((candidate) => candidate.key === modelKey);
    const value = pick ? toSlotValue(slot, pick) : null;

    if (!value) {
      const problem = models.some((candidate) => candidate.key === modelKey)
        ? `"${modelKey}" cannot be the ${slot.label} of ${values.model.name}`
        : `model key "${modelKey}" is not installed`;

      throw new Error(`components.${slotKey}: ${problem}. Choices: ${listModelKeys(fitting)}.`);
    }

    next = { ...next, [slot.key]: value };
  }

  return next;
};

/** Pick the first installed model for the first required slot that is unfilled; null when none can be filled. */
const withFirstRequiredComponent = (
  values: GenerateWidgetValues,
  models: readonly ModelConfig[]
): GenerateWidgetValues | null => {
  const ctx = getComponentPolicyContext(values.model, values);

  for (const slot of getComponentSectionPolicy(values.model, values).slots) {
    const current = ctx.selectedComponents[slot.key];

    if (!slot.required?.(ctx) || (current && (!slot.filter || slot.filter(current as ModelConfig, ctx)))) {
      continue;
    }

    for (const candidate of models) {
      const value = fitsSlot(slot, candidate, ctx) ? toSlotValue(slot, candidate) : null;

      if (value) {
        return { ...values, [slot.key]: value };
      }
    }
  }

  return null;
};

const withLoras = (
  values: GenerateWidgetValues,
  loras: GraphExportCase['loras'],
  models: readonly ModelConfig[]
): GenerateWidgetValues => {
  if (loras.length === 0) {
    return values;
  }

  const { model } = values;

  if (!isLoraSupported(model)) {
    throw new Error(`${model.name} does not load LoRAs.`);
  }

  const compatible = models.filter(isLoraModelConfig).filter((lora) => isLoraCompatibleWithModel(lora, model));

  return {
    ...values,
    loras: loras.map(({ key, weight }, index) => {
      const lora = compatible.find((candidate) => candidate.key === key);

      if (!lora) {
        const choices = listModelKeys(compatible);

        throw new Error(`loras[${index}]: "${key}" is not an installed LoRA for ${model.name}. Choices: ${choices}.`);
      }

      return { isEnabled: true, model: lora, weight };
    }),
  };
};

/** Adding the images to the Reference Images section, which types each one for the model as it joins the list. */
const withReferenceImages = (
  values: GenerateWidgetValues,
  input: GraphExportCase,
  models: readonly ModelConfig[]
): GenerateWidgetValues => {
  if (input.referenceImages.length === 0) {
    return values;
  }

  const { model } = values;

  if (!isReferenceImageSupported(model)) {
    const format = getMissingReferenceImageFormat(model);

    throw new Error(
      `reference_images: ${model.name} takes no reference images${format ? ` in this format; they need ${format}` : ''}.`
    );
  }

  return {
    ...values,
    referenceImages: withAddedReferenceImages(
      model,
      models,
      values.referenceImages,
      input.referenceImages.map((imageName, index) => ({
        id: `reference_image_${index + 1}`,
        // Cases name images only; the output size stands in for dimensions that reach metadata, not inference.
        image: { original: { image: { height: input.height, image_name: imageName, width: input.width } } },
      }))
    ),
  };
};

/** The widget reconciles after every commit; reconciliation can pick a FLUX.2 component source. */
const reconcile = (values: GenerateWidgetValues, models: readonly ModelConfig[]): GenerateWidgetValues => {
  const resolved = resolveGenerateWidgetValues({ models, storedValues: values });

  if (!resolved) {
    throw new Error('Generate cannot resolve settings without the capability table.');
  }

  return resolved.values;
};

/** Report a requested value the widget would not keep instead of compiling a graph nobody asked for. */
const assertRequestedValuesKept = (values: GenerateWidgetValues, input: GraphExportCase): void => {
  for (const [key, requested] of Object.entries(input.settings)) {
    const kept = values[key as keyof GenerateWidgetValues];

    if (!areJsonValuesStructurallyEqual(kept, requested)) {
      throw new Error(
        `settings.${key}: Generate does not keep ${JSON.stringify(requested)}; it holds ${JSON.stringify(kept)}.`
      );
    }
  }

  if ('scheduler' in input.settings && coerceSchedulerForGraph(values.model, values.scheduler) !== values.scheduler) {
    const options = getSchedulerOptions(values.model).map((option) => option.value);

    throw new Error(
      `settings.scheduler: ${values.model.name} does not use "${values.scheduler}". Options: ${listKeys(options)}.`
    );
  }

  for (const [key, requested] of Object.entries(input.components)) {
    const kept = values[key as keyof GenerateWidgetValues] as ComponentModelConfig | null;

    if (kept?.key !== requested) {
      throw new Error(
        `components.${key}: Generate replaces "${requested}" with ${kept ? `"${kept.key}"` : 'nothing'}.`
      );
    }
  }
};

const getControlLayers = (
  input: GraphExportCase,
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): ControlLayerGraphInput[] => {
  // A layer's kind is the one its model can drive on this base, the kind a user switches the layer to.
  const getKind = (candidate: ModelConfig) =>
    candidate.base === model.base
      ? CONTROL_ADAPTER_KINDS.find(
          (kind) => isControlKindSupportedForBase(model.base, kind) && isControlModelUsableForKind(candidate, kind)
        )
      : undefined;

  return input.control.map((layer, index) => {
    const adapter = models.find((candidate) => candidate.key === layer.modelKey);
    const kind = adapter ? getKind(adapter) : undefined;

    if (!adapter || !kind) {
      const choices = listModelKeys(models.filter((candidate) => getKind(candidate) !== undefined));
      const problem = `"${layer.modelKey}" cannot drive a control layer of ${model.name}`;

      throw new Error(`control[${index}]: ${problem}. Choices: ${choices}.`);
    }

    return {
      beginEndStepPct: [layer.begin, layer.end],
      // A new ControlNet layer starts "balanced", which the compiler sends for null; other kinds carry no mode.
      controlMode: null,
      id: `control_${index + 1}`,
      imageName: layer.imageName,
      kind,
      model: {
        base: adapter.base,
        key: adapter.key,
        name: adapter.name,
        type: adapter.type,
        ...(typeof adapter.hash === 'string' ? { hash: adapter.hash } : {}),
      },
      modelCondInChannels: typeof adapter.cond_in_channels === 'number' ? adapter.cond_in_channels : null,
      weight: layer.weight,
    };
  });
};

const assertModeInputs = (input: GraphExportCase): void => {
  if (input.mode === 'txt2img' && (input.initImage || input.control.length > 0)) {
    throw new Error('txt2img cases take no init_image or control; use img2img or controlnet.');
  }

  if (input.mode === 'img2img' && !input.initImage) {
    throw new Error('img2img cases need init_image.');
  }

  if (input.mode === 'controlnet' && input.control.length === 0) {
    throw new Error('controlnet cases need at least one control entry.');
  }
};

/** Throws a message naming the offending field and its valid choices when the case cannot compile. */
export const compileGraphExportCase = (
  input: GraphExportCase,
  env: GraphExportEnvironment
): CompiledGraphExportCase => {
  assertModeInputs(input);

  const { models } = env;
  let values = withCaseValues(getModelDefaultValues(getSelectableModel(input.modelKey, models), models), input);
  values = withSettingsOverrides(values, input.settings);
  values = withComponentPicks(values, input.components, models);
  values = withLoras(values, input.loras, models);
  values = reconcile(withReferenceImages(values, input, models), models);

  // Filling one slot can satisfy or require another; the bound stops a reconciliation that keeps undoing a pick.
  for (let pass = 0; pass < GENERATE_COMPONENT_VALUE_KEYS.length; pass += 1) {
    const filled = withFirstRequiredComponent(values, models);

    if (!filled) {
      break;
    }

    values = reconcile(filled, models);
  }

  assertRequestedValuesKept(values, input);

  // No expansion: the UI expands dynamic prompts through the backend before it queues.
  const { effectivePrompts, submission } = planGenerateSubmission(values);

  if (hasDynamicPromptSyntax(effectivePrompts.positivePrompt)) {
    throw new Error(
      'positive_prompt has dynamic prompt syntax, which only the backend expands; export each expansion.'
    );
  }

  const compiled =
    input.mode === 'txt2img'
      ? compileGenerateGraph(values, values.model, env.destination, env.projectSettings, env.randDevice)
      : compileCanvasGraph({
          bbox: { height: input.height, width: input.width, x: 0, y: 0 },
          compositeImageName: input.initImage?.imageName ?? null,
          compositing: env.canvas.compositing,
          controlLayers: getControlLayers(input, values.model, models),
          destination: env.destination,
          mode: input.initImage ? 'img2img' : 'txt2img',
          model: values.model,
          projectSettings: env.projectSettings,
          randDevice: env.randDevice,
          scaling: env.canvas.scaling,
          settings: values,
          strength: input.initImage?.strength ?? env.canvas.strength,
        });

  const stable = stabilizeBackendGraphNodeIds(compiled.backendGraph, {
    negativePromptNodeId: compiled.negativePromptNodeId,
    positivePromptNodeId: compiled.positivePromptNodeId,
    seedNodeId: compiled.seedNodeId,
  });

  return {
    components: Object.fromEntries(
      GENERATE_COMPONENT_VALUE_KEYS.flatMap((key) => {
        const component = values[key];

        return component ? [[key, component.key]] : [];
      })
    ),
    graph: { ...stable.graph, id: input.id },
    submission: { ...submission, ...stable.nodeIds },
  };
};
