/**
 * The record write gate. The recall contracts check every key a graph writes into `core_metadata`; this gate checks the
 * opposite blind spot: a setting that changes the compiled graph but is not recorded, so recall cannot restore it.
 *
 * Every Generate setting is varied, one at a time, from each coverage fixture, from the Canvas modes of each base's
 * first shape, and from supplier switches (own components against a component source). Whenever the varied graph
 * differs from the baseline in what the run reads, the record must differ too. A setting that never reaches the graph
 * passes without a trace, so a second test holds every setting to changing some fixture graph unless it is listed as
 * graph-inert.
 */

import type {
  ComponentModelConfig,
  GenerateLora,
  GenerateModelConfig,
  GenerateSettings,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';
import type {
  ComponentSlotPolicy,
  GenerateComponentValueKey,
  SupportedGenerateBase,
} from '@features/generation/core/baseGenerationPolicies';
import type { CompileCanvasGraphInput } from '@features/generation/core/canvas/types';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import {
  getComponentSectionPolicy,
  getFlux2ComponentSource,
  getGenerationValidationReasons,
} from '@features/generation/core/baseGenerationPolicies';
import { resolveCanvasProcessingSize } from '@features/generation/core/canvas/canvasProcessingSize';
import { getCanvasValidationReasons } from '@features/generation/core/canvas/compileCanvasGraph';
import { canvasModesFor } from '@features/generation/core/canvas/compileCanvasGraph.testing';
import {
  buildContext,
  candidatesForSlot,
  generateGraphCases,
  satisfiedSettingsFor,
} from '@features/generation/core/graphCoverage.testing';
import { compileGenerateGraph } from '@features/generation/graph';
import {
  getGenerationDimensions,
  getGenerationUiPolicy,
  getSchedulerOptions,
  isVaeCompatibleWithGenerateModel,
  withAddedReferenceImages,
} from '@features/generation/settings';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { describe, expect, it } from 'vitest';

import {
  type CanvasRecordCase,
  canvasRecordCases,
  canvasSettingsOf,
  compileCanvasRecordCase,
  type GenerateRecordCase,
  generateRecordCases,
  PROJECT_SETTINGS,
} from './imageRecordCases.testing';
import { recordOf } from './recallContract.testing';

type Json = Record<string, unknown>;
type Setting = keyof GenerateSettings;

interface GateCase {
  label: string;
  /** Set for a Canvas case: its mode, frame and compile extras, which hold for every varied setting. */
  canvas?: CanvasRecordCase;
  model: GenerateModelConfig;
  settings: GenerateSettings;
}

interface VariationContext {
  model: GenerateModelConfig;
  settings: GenerateSettings;
}

/** Candidate values for one setting, each different from the current one; validation may refuse some of them. */
type Variation = (context: VariationContext) => unknown[];

interface Compiled {
  graph: NormalizedGraph;
  record: Json;
}

interface NormalizedNode {
  type: unknown;
  fields: Json;
}

interface NormalizedGraph {
  nodes: NormalizedNode[];
  edges: Json[];
}

/**
 * Settings the record may leave out although they change the graph. Each entry names what the run does not need
 * recalled and why; one that suppresses nothing in its base's test fails as stale.
 */
interface Exception {
  /** The base whose test applies the entry and reports it stale. */
  base: GateBase;
  setting: Setting;
  /** Matches case labels; every case when absent. */
  cases?: RegExp;
  reason: string;
}

type GateBase = SupportedGenerateBase | 'external';

const EXCEPTIONS: readonly Exception[] = [];

/**
 * Settings the record holds only where the graph reads them. The VAE precision reaches the graph through an encode or
 * a decode; PiD replaces the decode, so a PiD run with no input to encode never used it.
 */
const PHANTOM_RECORD_SETTINGS: readonly Setting[] = ['vaePrecision'];

// ---------------------------------------------------------------------------------------------------------------
// Graph and record normalization

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/**
 * What a run reads from the graph: no graph id, no metadata node or its edges, no literal an edge overwrites at run
 * time (`_set_node_inputs` in graph_execution_runtime.py; collect and iterate take theirs from edges only), node ids by
 * insertion order since `createId` ids are random.
 */
const normalizeGraph = (backendGraph: CompiledGraph): NormalizedGraph => {
  const kept = Object.values(clone(backendGraph.nodes)).filter((node) => node.type !== 'core_metadata');
  const ids = new Map(kept.map((node, index) => [node.id, `n${index}`]));
  const edges = backendGraph.edges.filter((edge) => ids.has(edge.source.node_id) && ids.has(edge.destination.node_id));
  const fed = new Set(edges.map((edge) => `${edge.destination.node_id}\0${edge.destination.field}`));

  return {
    edges: edges.map((edge) => ({
      destination: { field: edge.destination.field, node: ids.get(edge.destination.node_id) },
      source: { field: edge.source.field, node: ids.get(edge.source.node_id) },
      type: edge.type ?? 'default',
    })),
    nodes: kept.map(({ id, type, ...fields }) => {
      const read: Json = {};

      if (type !== 'collect' && type !== 'iterate') {
        for (const [field, value] of Object.entries(fields)) {
          if (!fed.has(`${id}\0${field}`)) {
            read[field] = value;
          }
        }
      }

      return { fields: read, type };
    }),
  };
};

type CompiledGraph = ReturnType<typeof compileGenerateGraph>['backendGraph'];

/** A compile that validation accepted but the builder still threw on: policy and builder disagree. */
interface Drift {
  drift: string;
}

/**
 * Null where validation refused the settings; a `Drift` where it did not. The Canvas compile checks the shared generate
 * reasons, plus reasons of its own for PiD outside txt2img and img2img, which the contract also switches off.
 */
const compileCase = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  canvas?: CanvasRecordCase
): Compiled | Drift | null => {
  const effective = canvas ? canvasSettingsOf(canvas, settings) : settings;

  // The Canvas compile refuses on the shared reasons and on its own; the first is the one it throws.
  const refused = (): boolean => {
    if (getGenerationValidationReasons(model, effective).length > 0) {
      return true;
    }

    if (!canvas) {
      return false;
    }

    const { compile = {}, mode } = canvas;
    const { height, width } = canvas.frame ?? effective;
    const bbox = { height, width, x: 0, y: 0 };

    return (
      getCanvasValidationReasons(
        {
          ...compile,
          bbox,
          compositeImageName: 'compositeImageName' in compile ? compile.compositeImageName : 'canvas-composite.png',
          maskImageName:
            'maskImageName' in compile ? compile.maskImageName : mode === 'inpaint' ? 'canvas-mask.png' : null,
          mode,
          model,
          settings: effective,
          strength: compile.strength ?? 0.6,
        } as CompileCanvasGraphInput,
        resolveCanvasProcessingSize(model, effective.pidMode, bbox, compile.scaling)
      ).length > 0
    );
  };

  try {
    const compiled = canvas
      ? compileCanvasRecordCase({ ...canvas, model }, settings)
      : compileGenerateGraph(effective, model, 'gallery', PROJECT_SETTINGS);

    return { graph: normalizeGraph(compiled.backendGraph), record: clone(recordOf(compiled, effective)) };
  } catch (error) {
    return refused() ? null : { drift: error instanceof Error ? error.message : String(error) };
  }
};

const isDrift = (result: Compiled | Drift | null): result is Drift => result !== null && 'drift' in result;

/** Where two graphs differ, as `type.field` for a changed field and `+type` / `-type` for a node only one has. */
const graphHint = (before: NormalizedGraph, after: NormalizedGraph): string => {
  const hints: string[] = [];

  if (before.nodes.length !== after.nodes.length) {
    const counts = new Map<string, number>();

    for (const { type } of after.nodes) {
      counts.set(String(type), (counts.get(String(type)) ?? 0) + 1);
    }

    for (const { type } of before.nodes) {
      counts.set(String(type), (counts.get(String(type)) ?? 0) - 1);
    }

    for (const [type, count] of counts) {
      if (count !== 0) {
        hints.push(`${count > 0 ? '+' : '-'}${type}`);
      }
    }
  } else {
    before.nodes.forEach((node, index) => {
      const other = after.nodes[index]!;

      if (node.type !== other.type) {
        hints.push(`${String(node.type)} -> ${String(other.type)}`);
        return;
      }

      for (const field of new Set([...Object.keys(node.fields), ...Object.keys(other.fields)])) {
        if (!areJsonValuesStructurallyEqual(node.fields[field], other.fields[field])) {
          hints.push(`${String(node.type)}.${field}`);
        }
      }
    });
  }

  if (hints.length === 0) {
    hints.push('edges');
  }

  return hints.slice(0, 4).join(', ');
};

const show = (value: unknown): string => {
  const text =
    value && typeof value === 'object' && 'key' in value && typeof value.key === 'string'
      ? `model ${value.key}`
      : (JSON.stringify(value) ?? 'undefined');

  return text.length > 70 ? `${text.slice(0, 67)}...` : text;
};

// ---------------------------------------------------------------------------------------------------------------
// Variation table

const others = <T>(current: T, all: readonly T[]): T[] => all.filter((value) => value !== current);

/** The neighbours of a number on both sides, inside the bounds where the setting has them. */
const stepped =
  (delta: number, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) =>
  (current: number): number[] =>
    [current + delta, current - delta].filter((value) => value >= min && value <= max && value !== current);

const nullableNumber =
  (...values: number[]) =>
  (current: number | null): Array<number | null> =>
    current === null ? values : [null, ...values.filter((value) => value !== current).slice(0, 1)];

const REFERENCE_ASSET = { original: { image: { height: 1024, image_name: 'gate-reference.png', width: 1024 } } };

const REFINER: MainModelConfig = { base: 'sdxl-refiner', key: 'gate-refiner', name: 'Gate Refiner', type: 'main' };

/** Per policy-slot candidate lists are large and filtered lazily; cache the unfiltered ones. */
const slotCandidates = new Map<string, ReturnType<typeof candidatesForSlot>>();

/**
 * A main model offered as a component source where the policy has no slot for it: FLUX.2 keeps the source it takes from
 * the installed pipelines in the settings without a picker. The candidates are the model's own base, its variant first;
 * the compile decides whether one is accepted.
 */
const UNLISTED_SOURCE_SLOT: ComponentSlotPolicy = {
  key: 'componentSourceModel',
  label: 'Component source',
  modelTypes: ['main'],
  valueKind: 'main',
};

/** SD takes its VAE override without a slot in the policy; the graph builder sends the compatible ones. */
const UNLISTED_VAE_SLOT: ComponentSlotPolicy = {
  key: 'vae',
  label: 'VAE',
  modelTypes: ['vae'],
  valueKind: 'vae',
};

/**
 * The accepted candidates of a slot for the model, most fitting first; for a slot the policy does not list, those the
 * rule that sends them to the graph accepts.
 */
const acceptedCandidates = (
  model: MainModelConfig,
  settings: GenerateSettings,
  key: GenerateComponentValueKey
): ComponentModelConfig[] => {
  const { slots } = getComponentSectionPolicy(model, settings);
  const listed = slots.find((candidate) => candidate.key === key);
  const slot =
    listed ??
    (key === 'componentSourceModel'
      ? UNLISTED_SOURCE_SLOT
      : key === 'vae' && getGenerationUiPolicy(model).sdVaeVisible
        ? UNLISTED_VAE_SLOT
        : undefined);

  if (!slot) {
    return [];
  }

  const cacheKey = `${slot.modelTypes.join()}|${slot.valueKind}`;
  const all = slotCandidates.get(cacheKey) ?? candidatesForSlot(slot);
  slotCandidates.set(cacheKey, all);

  if (slot === UNLISTED_VAE_SLOT) {
    return all.filter((candidate) => isVaeCompatibleWithGenerateModel(model, candidate as VaeModelConfig));
  }

  if (!listed) {
    return all
      .filter((candidate) => candidate.base === model.base)
      .sort((left, right) => Number(right.variant === model.variant) - Number(left.variant === model.variant));
  }

  const context = buildContext(model, settings, slots);

  return all.filter((candidate) => !slot.filter || slot.filter(candidate, context));
};

/** Another candidate the slot accepts (up to two), and null when a component is selected. */
const componentVariation =
  (key: GenerateComponentValueKey): Variation =>
  ({ model, settings }) => {
    if (model.type !== 'main') {
      return [];
    }

    const current = settings[key] as ComponentModelConfig | null;
    const picks = acceptedCandidates(model, settings, key)
      .filter((candidate) => candidate.key !== current?.key)
      .slice(0, 2);

    return current ? [null, ...picks] : picks;
  };

const flip = (current: boolean): boolean[] => [!current];

const VARIATIONS = {
  // Prompts and templates.
  batchCount: ({ settings }) => stepped(1, 1)(settings.batchCount),
  dynamicPromptsCombinatorial: ({ settings }) => flip(settings.dynamicPromptsCombinatorial),
  dynamicPromptsEnabled: ({ settings }) => flip(settings.dynamicPromptsEnabled),
  dynamicPromptsMaxPrompts: ({ settings }) => stepped(3, 1)(settings.dynamicPromptsMaxPrompts),
  dynamicPromptsResample: ({ settings }) => flip(settings.dynamicPromptsResample),
  dynamicPromptsSampleSeed: ({ settings }) => stepped(7, 0)(settings.dynamicPromptsSampleSeed),
  dynamicPromptsSeedBehaviour: ({ settings }) =>
    others(settings.dynamicPromptsSeedBehaviour, ['per-iteration', 'per-image']),
  expandPromptModelKey: ({ settings }) => (settings.expandPromptModelKey === null ? ['gate-llm'] : [null]),
  imageToPromptModelKey: ({ settings }) => (settings.imageToPromptModelKey === null ? ['gate-vlm'] : [null]),
  negativePrompt: ({ settings }) => [`${settings.negativePrompt} gate`],
  negativePromptEnabled: ({ settings }) => flip(settings.negativePromptEnabled),
  negativePromptHeightPx: ({ settings }) => stepped(20, 56, 240)(settings.negativePromptHeightPx),
  positivePrompt: ({ settings }) => [`${settings.positivePrompt} gate`],
  positivePromptHeightPx: ({ settings }) => stepped(20, 96, 360)(settings.positivePromptHeightPx),
  promptTemplate: ({ settings }) =>
    settings.promptTemplate
      ? [null]
      : [{ id: 'gate', name: 'Gate', negativePrompt: 'blurry', positivePrompt: 'gate, {prompt}' }],
  promptTemplateViewMode: ({ settings }) => flip(settings.promptTemplateViewMode),

  // Size.
  aspectRatioId: ({ settings }) => others(settings.aspectRatioId, ['Free', '1:1', '16:9', '3:4']),
  aspectRatioIsLocked: ({ settings }) => flip(settings.aspectRatioIsLocked),
  aspectRatioValue: ({ settings }) => [settings.aspectRatioValue * 1.25],
  height: ({ model, settings }) =>
    stepped(getGenerationDimensions(model, settings.pidMode).grid, 64, 4096)(settings.height),
  width: ({ model, settings }) =>
    stepped(getGenerationDimensions(model, settings.pidMode).grid, 64, 4096)(settings.width),

  // Sampling.
  cfgRescaleMultiplier: ({ settings }) => stepped(0.3, 0, 0.99)(settings.cfgRescaleMultiplier),
  cfgScale: ({ model, settings }) => {
    const { guidanceMax, guidanceMin } = getGenerationUiPolicy(model);

    return stepped(0.5, guidanceMin, guidanceMax ?? Number.POSITIVE_INFINITY)(settings.cfgScale);
  },
  clipSkip: ({ model, settings }) => stepped(1, 0, getGenerationUiPolicy(model).clipSkipMax ?? 2)(settings.clipSkip),
  colorCompensation: ({ settings }) => flip(settings.colorCompensation),
  hiDiffusionEnabled: ({ settings }) => flip(settings.hiDiffusionEnabled),
  hiDiffusionManualRatios: ({ settings }) => (settings.hiDiffusionManualRatios ? [undefined] : [{ t1: 0.4, t2: 0.7 }]),
  hiDiffusionRauNetEnabled: ({ settings }) => flip(settings.hiDiffusionRauNetEnabled),
  hiDiffusionT1Ratio: ({ settings }) => nullableNumber(0.35, 0.6)(settings.hiDiffusionT1Ratio),
  hiDiffusionT2Ratio: ({ settings }) => nullableNumber(0.55, 0.8)(settings.hiDiffusionT2Ratio),
  hiDiffusionWindowAttentionEnabled: ({ settings }) => flip(settings.hiDiffusionWindowAttentionEnabled),
  scheduler: ({ model, settings }) =>
    getSchedulerOptions(model as Parameters<typeof getSchedulerOptions>[0])
      .map((option) => option.value)
      .filter((value) => value !== settings.scheduler)
      .slice(0, 3),
  seamlessXAxis: ({ settings }) => flip(settings.seamlessXAxis),
  seamlessYAxis: ({ settings }) => flip(settings.seamlessYAxis),
  seed: ({ settings }) => [settings.seed + 1],
  seedMode: ({ settings }) => others(settings.seedMode, ['random', 'fixed', 'increment', 'decrement']),
  steps: ({ settings }) => stepped(3, 1)(settings.steps),

  // Identity: the model key is read by the project, the model by the graph.
  modelKey: ({ settings }) => [`${settings.modelKey}-gate`],

  // Components.
  animaInpaintAdapterEnabled: ({ settings }) => flip(settings.animaInpaintAdapterEnabled),
  animaInpaintAdapterModel: componentVariation('animaInpaintAdapterModel'),
  animaInpaintAdapterWeight: ({ settings }) => stepped(0.5, -10, 10)(settings.animaInpaintAdapterWeight),
  clipEmbedModel: componentVariation('clipEmbedModel'),
  clipGEmbedModel: componentVariation('clipGEmbedModel'),
  clipLEmbedModel: componentVariation('clipLEmbedModel'),
  componentSourceModel: componentVariation('componentSourceModel'),
  flux3PromptUpsamplerEnabled: ({ settings }) => flip(settings.flux3PromptUpsamplerEnabled),
  flux3PromptUpsamplerModel: componentVariation('flux3PromptUpsamplerModel'),
  gemma2EncoderModel: componentVariation('gemma2EncoderModel'),
  ideogram4UnconditionalModel: componentVariation('ideogram4UnconditionalModel'),
  mistralEncoderModel: componentVariation('mistralEncoderModel'),
  pidDecoderModel: componentVariation('pidDecoderModel'),
  qwen35EncoderModel: componentVariation('qwen35EncoderModel'),
  qwen3EncoderModel: componentVariation('qwen3EncoderModel'),
  qwen3VLEncoderModel: componentVariation('qwen3VLEncoderModel'),
  qwen3VLVisionTowerModel: componentVariation('qwen3VLVisionTowerModel'),
  qwenVLEncoderModel: componentVariation('qwenVLEncoderModel'),
  t5EncoderModel: componentVariation('t5EncoderModel'),
  vae: componentVariation('vae'),
  vaePrecision: ({ settings }) => [settings.vaePrecision === 'fp32' ? 'fp16' : 'fp32'],
  wanLowNoiseModel: componentVariation('wanLowNoiseModel'),
  wanT5EncoderModel: componentVariation('wanT5EncoderModel'),

  // Adapters.
  loras: ({ model, settings }) => {
    const added: GenerateLora = {
      isEnabled: true,
      model: { base: model.base, key: `${model.base}-gate-lora`, name: 'Gate LoRA', type: 'lora' },
      weight: 0.8,
    };
    const [first, ...rest] = settings.loras;

    return first
      ? [
          [...settings.loras, added],
          [],
          [{ ...first, weight: first.weight + 0.2 }, ...rest],
          [{ ...first, isEnabled: !first.isEnabled }, ...rest],
        ]
      : [[added]];
  },
  referenceImages: ({ model, settings }) => {
    const [first, ...rest] = settings.referenceImages;
    const added = withAddedReferenceImages(model, [], settings.referenceImages, [
      { id: 'gate-reference', image: REFERENCE_ASSET },
    ]);

    return first ? [added, [], [{ ...first, isEnabled: !first.isEnabled }, ...rest]] : [added];
  },

  // Family tuning.
  fluxDypeExponent: ({ settings }) => stepped(1, 0, 1000)(settings.fluxDypeExponent),
  fluxDypePreset: ({ settings }) => others(settings.fluxDypePreset, ['off', 'manual', 'auto', 'area', '4k']),
  fluxDypeScale: ({ settings }) => stepped(1, 0, 8)(settings.fluxDypeScale),
  ideogram4ColorPalette: ({ settings }) =>
    settings.ideogram4ColorPalette.length > 0 ? [[], [...settings.ideogram4ColorPalette, 'gate blue']] : [['gate red']],
  ideogram4GuidanceScale: ({ settings }) => nullableNumber(2.5, 4.5)(settings.ideogram4GuidanceScale),
  ideogram4Mu: ({ settings }) => nullableNumber(0.5, 1.5)(settings.ideogram4Mu),
  ideogram4SamplerPreset: ({ settings }) =>
    others(settings.ideogram4SamplerPreset, ['V4_QUALITY_48', 'V4_DEFAULT_20', 'V4_TURBO_12']),
  ideogram4Steps: ({ settings }) => nullableNumber(10, 30)(settings.ideogram4Steps),
  krea2ReferenceKvCache: ({ settings }) => flip(settings.krea2ReferenceKvCache),
  krea2RebalanceEnabled: ({ settings }) => flip(settings.krea2RebalanceEnabled),
  krea2RebalanceMultiplier: ({ settings }) => stepped(1, 0, 20)(settings.krea2RebalanceMultiplier),
  // Changes the first of the fixed number of weights; a further one is refused.
  krea2RebalanceWeights: ({ settings }) => {
    const [first = '1', ...rest] = settings.krea2RebalanceWeights.split(',');

    return [[String(Number(first) + 0.5), ...rest].join(',')];
  },
  krea2SeedVarianceEnabled: ({ settings }) => flip(settings.krea2SeedVarianceEnabled),
  krea2SeedVarianceRandomizePercent: ({ settings }) => stepped(10, 1, 100)(settings.krea2SeedVarianceRandomizePercent),
  krea2SeedVarianceStrength: ({ settings }) => stepped(0.2, 0, 2)(settings.krea2SeedVarianceStrength),
  mossImageUncondSkipLayers: ({ settings }) =>
    settings.mossImageUncondSkipLayers.length > 0
      ? [[], [...settings.mossImageUncondSkipLayers, Math.max(...settings.mossImageUncondSkipLayers) + 1]]
      : [[2], [2, 4]],
  pidMode: ({ settings }) => others(settings.pidMode, ['off', 'fit', 'native']),
  pidSteps: ({ settings }) => stepped(2, 1, 16)(settings.pidSteps),
  qwenImageEncoderQuantization: ({ settings }) =>
    others(settings.qwenImageEncoderQuantization, ['none', 'int8', 'nf4']),
  qwenImageShift: ({ settings }) => nullableNumber(2, 4)(settings.qwenImageShift),
  wanGuidanceScaleLowNoise: ({ settings }) => nullableNumber(2, 4)(settings.wanGuidanceScaleLowNoise),
  zImageSeedVarianceEnabled: ({ settings }) => flip(settings.zImageSeedVarianceEnabled),
  zImageSeedVarianceRandomizePercent: ({ settings }) =>
    stepped(10, 1, 100)(settings.zImageSeedVarianceRandomizePercent),
  zImageSeedVarianceStrength: ({ settings }) => stepped(0.2, 0, 2)(settings.zImageSeedVarianceStrength),
  zImageShift: ({ settings }) => nullableNumber(2, 4)(settings.zImageShift),

  // Refiner.
  refinerCfgScale: ({ settings }) => stepped(1, 1)(settings.refinerCfgScale),
  // From a set refiner, a second one first: clearing or adding one also changes the other refiner nodes.
  refinerModel: ({ settings }) =>
    settings.refinerModel ? [{ ...REFINER, key: 'gate-refiner-2', name: 'Gate Refiner 2' }, null] : [REFINER],
  refinerNegativeAestheticScore: ({ settings }) => stepped(1, 1, 10)(settings.refinerNegativeAestheticScore),
  refinerPositiveAestheticScore: ({ settings }) => stepped(1, 1, 10)(settings.refinerPositiveAestheticScore),
  refinerScheduler: ({ settings }) => others(settings.refinerScheduler, ['euler', 'dpmpp_2m', 'ddim']),
  refinerStart: ({ settings }) => stepped(0.1, 0, 1)(settings.refinerStart),
  refinerSteps: ({ settings }) => stepped(4, 1)(settings.refinerSteps),

  // Hosted providers.
  externalImageSize: ({ model, settings }) => {
    const sizes =
      model.type === 'external_image_generator'
        ? (model.capabilities?.resolution_presets ?? []).map((preset) => preset.image_size)
        : [];

    return [...(settings.externalImageSize === null ? [] : [null]), ...others(settings.externalImageSize, sizes)];
  },
  geminiTemperature: ({ settings }) => nullableNumber(0.4, 1.4)(settings.geminiTemperature),
  geminiThinkingLevel: ({ settings }) => others(settings.geminiThinkingLevel, [null, 'minimal', 'high'] as const),
  openaiBackground: ({ settings }) => others(settings.openaiBackground, ['auto', 'transparent', 'opaque']),
  openaiInputFidelity: ({ settings }) => others(settings.openaiInputFidelity, [null, 'low', 'high'] as const),
  openaiQuality: ({ settings }) => others(settings.openaiQuality, ['auto', 'high', 'medium', 'low']),
  seedreamOptimizePrompt: ({ settings }) => flip(settings.seedreamOptimizePrompt),
  seedreamWatermark: ({ settings }) => flip(settings.seedreamWatermark),
} satisfies Record<Setting, Variation>;

// ---------------------------------------------------------------------------------------------------------------
// Cases

seedArchitectureCapabilities();

const gateBaseOf = ({ model }: { model: GenerateModelConfig }): GateBase =>
  model.type === 'external_image_generator' ? 'external' : (model.base as GateBase);

/** The shared record cases, built once on first use (not in a hook, so a filtered run pays only for what it runs). */
let sharedCases: { canvas: CanvasRecordCase[]; generate: GenerateRecordCase[] } | undefined;

const shared = () => (sharedCases ??= { canvas: canvasRecordCases(), generate: generateRecordCases() });

/** The base's cases with its required components only, beside the shared ones: the shapes tuned, and the extras. */
const baselineCases = (gateBase: GateBase): GateCase[] => [
  ...generateGraphCases
    .filter(({ base }) => base === gateBase)
    .map(({ base, shape }): GateCase => {
      const satisfied = satisfiedSettingsFor(base, shape);

      return {
        label: `${base} / ${shape.label} (required components)`,
        model: satisfied.model,
        settings: satisfied.settings,
      };
    }),
  ...shared().generate.filter((fixture) => gateBaseOf(fixture) === gateBase),
];

/**
 * The Canvas modes of each base's first shape with its required components only, beside the shared Canvas cases.
 */
const canvasCases = (gateBase: GateBase): GateCase[] => [
  ...generateGraphCases
    .filter(({ base }, index, all) => base === gateBase && all.findIndex((other) => other.base === base) === index)
    .flatMap(({ base, shape }) => {
      const satisfied = satisfiedSettingsFor(base, shape);

      return canvasModesFor(satisfied.model, satisfied.settings).map((mode): GateCase => {
        const label = `${base} / ${shape.label} (required components) / canvas ${mode}`;

        return {
          canvas: { label, mode, model: satisfied.model, settings: satisfied.settings },
          label,
          model: satisfied.model,
          settings: satisfied.settings,
        };
      });
    }),
  ...shared()
    .canvas.filter((fixture) => gateBaseOf(fixture) === gateBase)
    .map((fixture): GateCase => ({
      canvas: fixture,
      label: fixture.label,
      model: fixture.model,
      settings: fixture.settings,
    })),
];

/**
 * The supplier switch: the same model with its own encoders and VAE, and with a component source in their place. The
 * source makes the slots it supplies optional; those are cleared. A model whose source never reaches the graph (no
 * source, or a bundled main) has no pair.
 */
const supplierCases = (failures: string[], gateBase: GateBase): Array<{ own: GateCase; sourced: GateCase }> =>
  generateGraphCases
    .filter(({ base }) => base === gateBase)
    .flatMap(({ base, shape }) => {
      const { model, settings } = satisfiedSettingsFor(base as SupportedGenerateBase, shape);
      const label = `${base} / ${shape.label}`;

      if (settings.componentSourceModel) {
        return [];
      }

      // Own VAE where the policy has the slot, the encoders being the required components already.
      const ownVae = acceptedCandidates(model, settings, 'vae')[0] as VaeModelConfig | undefined;
      const own: GenerateSettings = { ...settings, vae: ownVae ?? settings.vae };
      const ownCompiled = compileCase(model, own);

      if (isDrift(ownCompiled)) {
        failures.push(
          `\`${label} (own components)\` does not compile although validation accepts it: ${ownCompiled.drift}`
        );
      }

      if (!ownCompiled || isDrift(ownCompiled)) {
        return [];
      }

      for (const source of acceptedCandidates(model, own, 'componentSourceModel').slice(0, 12)) {
        let sourced: GenerateSettings = { ...own, componentSourceModel: source as MainModelConfig };
        const { slots } = getComponentSectionPolicy(model, sourced);
        const context = buildContext(model, sourced, slots);

        // What the source supplies is optional now: clear the slots no longer required.
        for (const slot of slots) {
          if (slot.key !== 'componentSourceModel' && sourced[slot.key] && !slot.required?.(context)) {
            sourced = { ...sourced, [slot.key]: null };
          }
        }

        const sourcedCompiled = compileCase(model, sourced);
        const withoutSource = compileCase(model, { ...sourced, componentSourceModel: null });

        for (const [what, result] of [
          ['component source', sourcedCompiled],
          ['without the source', withoutSource],
        ] as const) {
          if (isDrift(result)) {
            failures.push(
              `\`${label} (${what})\`: componentSourceModel ${show(source)} does not compile although validation accepts it: ${result.drift}`
            );
          }
        }

        // The source reaches the graph where leaving it out is refused or builds another graph; one a family ignores
        // (or a bundled main) is no supplier switch.
        if (
          sourcedCompiled &&
          !isDrift(sourcedCompiled) &&
          !isDrift(withoutSource) &&
          (!withoutSource || !areJsonValuesStructurallyEqual(withoutSource.graph, sourcedCompiled.graph))
        ) {
          return [
            {
              own: { label: `${label} (own components)`, model, settings: own },
              sourced: { label: `${label} (component source)`, model, settings: sourced },
            },
          ];
        }
      }

      // The policy offers a source (a slot, or the FLUX.2 source kept in the settings) but none built a pair.
      if (canTakeSource(model, settings)) {
        failures.push(`\`${label}\`: the policy takes a component source but no supplier pair formed`);
      }

      return [];
    });

/** Whether the policy lets the model take a component source: its own slot, or FLUX.2's implicit one. */
const canTakeSource = (model: MainModelConfig, settings: GenerateSettings): boolean =>
  getComponentSectionPolicy(model, settings).slots.some((slot) => slot.key === 'componentSourceModel') ||
  acceptedCandidates(model, settings, 'componentSourceModel').some(
    (candidate) =>
      getFlux2ComponentSource(model, { ...settings, componentSourceModel: candidate as MainModelConfig }) !== undefined
  );

// ---------------------------------------------------------------------------------------------------------------
// Gate

const GATE_BASES: GateBase[] = [...new Set(generateGraphCases.map(({ base }) => base)), 'external'];

describe('image record write gate', () => {
  it.each(GATE_BASES)('records every setting of %s that changes the compiled graph', (gateBase) => {
    const failures: string[] = [];
    const used = new Map<Exception, number>();
    const settingsKeys = Object.keys(VARIATIONS) as Setting[];
    const pairs = supplierCases(failures, gateBase);
    const cases = [
      ...baselineCases(gateBase),
      ...canvasCases(gateBase),
      ...pairs.flatMap(({ own, sourced }) => [own, sourced]),
    ];

    const excepted = (setting: Setting, label: string): boolean => {
      const entry = EXCEPTIONS.find(
        (candidate) =>
          candidate.base === gateBase &&
          candidate.setting === setting &&
          (!candidate.cases || candidate.cases.test(label))
      );

      if (entry) {
        used.set(entry, (used.get(entry) ?? 0) + 1);
      }

      return entry !== undefined;
    };

    const check = (
      label: string,
      setting: Setting,
      before: unknown,
      after: unknown,
      base: Compiled,
      varied: Compiled
    ): boolean => {
      // The converse for the settings a record may only hold while the run reads them: a record that moves with the
      // setting although no node reads it names a value the run never had.
      if (
        PHANTOM_RECORD_SETTINGS.includes(setting) &&
        areJsonValuesStructurallyEqual(base.graph, varied.graph) &&
        !areJsonValuesStructurallyEqual(base.record, varied.record)
      ) {
        failures.push(
          `\`${label}\`: ${setting} ${show(before)} -> ${show(after)} changes the record but no node the run reads`
        );
        return false;
      }

      if (
        areJsonValuesStructurallyEqual(base.graph, varied.graph) ||
        !areJsonValuesStructurallyEqual(base.record, varied.record)
      ) {
        return true;
      }

      if (!excepted(setting, label)) {
        failures.push(
          `\`${label}\`: ${setting} ${show(before)} -> ${show(after)} changes the graph (${graphHint(base.graph, varied.graph)}) but not the record`
        );
      }

      return false;
    };

    for (const { canvas, label, model, settings } of cases) {
      const base = compileCase(model, settings, canvas);

      if (!base || isDrift(base)) {
        failures.push(
          `\`${label}\` does not compile${base ? ` although validation accepts it: ${base.drift}` : ' (validation refuses it)'}`
        );
        continue;
      }

      for (const setting of settingsKeys) {
        const candidates = (VARIATIONS[setting] as Variation)({ model, settings }).filter(
          (value) => !areJsonValuesStructurallyEqual(value, settings[setting])
        );

        for (const value of candidates) {
          const varied = compileCase(model, { ...settings, [setting]: value }, canvas);

          if (isDrift(varied)) {
            failures.push(
              `\`${label}\`: ${setting} ${show(settings[setting])} -> ${show(value)} does not compile although validation accepts it: ${varied.drift}`
            );
            continue;
          }

          if (!varied) {
            continue;
          }

          if (!check(label, setting, settings[setting], value, base, varied)) {
            break;
          }
        }
      }
    }

    for (const { own, sourced } of pairs) {
      const before = compileCase(own.model, own.settings, own.canvas) as Compiled;
      const after = compileCase(sourced.model, sourced.settings, sourced.canvas) as Compiled;

      check(
        sourced.label,
        'componentSourceModel',
        'own components',
        sourced.settings.componentSourceModel,
        before,
        after
      );
    }

    for (const entry of EXCEPTIONS.filter((candidate) => candidate.base === gateBase)) {
      if (!used.has(entry)) {
        failures.push(`stale exception for ${entry.setting} (${entry.reason}) suppresses nothing`);
      }
    }

    expect(failures).toEqual([]);
  });
});

/**
 * Settings no compiled graph reads, so the gate has nothing to hold to the record: the queue carries them with the
 * batch, or only the UI reads them. One that does change a fixture graph fails as stale.
 */
const BATCH = 'The queue fills it from the batch when the graph runs; a graph literal it feeds is overwritten.';
const UI_ONLY = 'The panel reads it; no graph builder does.';

const GRAPH_INERT: ReadonlyArray<{ setting: Setting; reason: string }> = [
  { reason: BATCH, setting: 'batchCount' },
  { reason: BATCH, setting: 'dynamicPromptsCombinatorial' },
  { reason: BATCH, setting: 'dynamicPromptsEnabled' },
  { reason: BATCH, setting: 'dynamicPromptsMaxPrompts' },
  { reason: BATCH, setting: 'dynamicPromptsResample' },
  { reason: BATCH, setting: 'dynamicPromptsSampleSeed' },
  { reason: BATCH, setting: 'dynamicPromptsSeedBehaviour' },
  { reason: BATCH, setting: 'negativePrompt' },
  { reason: BATCH, setting: 'negativePromptEnabled' },
  { reason: BATCH, setting: 'positivePrompt' },
  { reason: BATCH, setting: 'promptTemplate' },
  { reason: BATCH, setting: 'promptTemplateViewMode' },
  { reason: BATCH, setting: 'seed' },
  { reason: BATCH, setting: 'seedMode' },
  { reason: UI_ONLY, setting: 'aspectRatioId' },
  { reason: UI_ONLY, setting: 'aspectRatioIsLocked' },
  { reason: UI_ONLY, setting: 'aspectRatioValue' },
  { reason: UI_ONLY, setting: 'hiDiffusionManualRatios' },
  { reason: UI_ONLY, setting: 'modelKey' },
  { reason: UI_ONLY, setting: 'expandPromptModelKey' },
  { reason: UI_ONLY, setting: 'imageToPromptModelKey' },
  { reason: UI_ONLY, setting: 'negativePromptHeightPx' },
  { reason: UI_ONLY, setting: 'positivePromptHeightPx' },
  {
    setting: 'externalImageSize',
    reason:
      'Picks among presets of one size; the provider node is asked for the width and height, which the preset shares.',
  },
];

/** The settings of `wanted` that change some fixture graph, each searched until its first change. */
const settingsChangingAGraph = (wanted: readonly Setting[]): Set<Setting> => {
  const unresolved = new Set(wanted);
  const reaching = new Set<Setting>();

  for (const gateBase of GATE_BASES) {
    if (unresolved.size === 0) {
      break;
    }

    for (const { canvas, model, settings } of [...baselineCases(gateBase), ...canvasCases(gateBase)]) {
      const base = compileCase(model, settings, canvas);

      if (!base || isDrift(base)) {
        continue;
      }

      for (const setting of unresolved) {
        const candidates = (VARIATIONS[setting] as Variation)({ model, settings }).filter(
          (value) => !areJsonValuesStructurallyEqual(value, settings[setting])
        );

        for (const value of candidates) {
          const varied = compileCase(model, { ...settings, [setting]: value }, canvas);

          if (varied && !isDrift(varied) && !areJsonValuesStructurallyEqual(base.graph, varied.graph)) {
            reaching.add(setting);
            unresolved.delete(setting);
            break;
          }
        }
      }
    }
  }

  return reaching;
};

const ALL_SETTINGS = Object.keys(VARIATIONS) as Setting[];
const INERT_SETTINGS = GRAPH_INERT.map(({ setting }) => setting);

describe('image record write gate reachability', () => {
  it('varies every setting that is not graph-inert into some fixture graph', () => {
    const reaching = settingsChangingAGraph(ALL_SETTINGS.filter((setting) => !INERT_SETTINGS.includes(setting)));

    expect(ALL_SETTINGS.filter((setting) => !INERT_SETTINGS.includes(setting) && !reaching.has(setting))).toEqual([]);
  });

  it('lists no graph-inert setting that changes a fixture graph', () => {
    const reaching = settingsChangingAGraph(INERT_SETTINGS);

    expect(
      GRAPH_INERT.filter(({ setting }) => reaching.has(setting)).map(({ setting, reason }) => `${setting} (${reason})`)
    ).toEqual([]);
  });
});
