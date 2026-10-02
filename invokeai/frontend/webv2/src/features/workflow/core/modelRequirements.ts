import type { FieldInputTemplate, InvocationTemplates, ProjectGraphState } from './types';

import { isLoraFieldCollectionEntry, isModelFieldType, toLoraFieldCollectionList } from './fields';
import { isInvocationNode } from './types';

/**
 * Extract exact models and template-described empty required slots; blank slots are normal workflow requirements,
 * not merely a fallback.
 */

export interface WorkflowModelIdentifier {
  key: string;
  hash?: string;
  name?: string;
  base?: string;
  type?: string;
}

export type WorkflowModelRequirement =
  | { kind: 'exact'; identifier: WorkflowModelIdentifier; label: string }
  | { kind: 'slot'; base: string | null; modelType: string | null; label: string };

export interface WorkflowModelRequirementSet {
  requirements: readonly WorkflowModelRequirement[];
  primaryBase: string | null;
}

// #region Labeling

/** Display labels for known model bases. Unknown bases fall back to a humanized form. */
const BASE_LABELS: Record<string, string> = {
  any: 'Any',
  anima: 'Anima',
  cogview4: 'CogView4',
  'ernie-image': 'ERNIE-Image',
  external: 'External',
  flux: 'FLUX',
  flux2: 'FLUX.2',
  'ideogram-4': 'Ideogram 4',
  'krea-2': 'Krea-2',
  'minimax-h3': 'MiniMax H3',
  'qwen-image': 'Qwen Image',
  'sd-1': 'SD 1.x',
  'sd-2': 'SD 2.x',
  'sd-3': 'SD 3.x',
  sdxl: 'SDXL',
  'sdxl-refiner': 'SDXL Refiner',
  unknown: 'Unknown',
  wan: 'Wan 2.2',
  'z-image': 'Z-Image',
};

/** Display labels for known model taxonomy types. Unknown types fall back to a humanized form. */
const MODEL_TYPE_LABELS: Record<string, string> = {
  clip_embed: 'CLIP embed',
  clip_vision: 'CLIP Vision',
  control_lora: 'Control LoRA',
  controlnet: 'ControlNet',
  embedding: 'embedding',
  external_image_generator: 'external image generator',
  flux_redux: 'FLUX Redux',
  gemma2_encoder: 'Gemma2 encoder',
  ip_adapter: 'IP Adapter',
  llava_onevision: 'LLaVA OneVision',
  lora: 'LoRA',
  main: 'checkpoint',
  mistral_encoder: 'Mistral encoder',
  onnx: 'ONNX model',
  pid_decoder: 'PID decoder',
  qwen3_encoder: 'Qwen3 encoder',
  qwen3_vl_encoder: 'Qwen3 VL encoder',
  qwen_vl_encoder: 'Qwen VL encoder',
  siglip: 'SigLIP',
  spandrel_image_to_image: 'upscaler',
  t2i_adapter: 'T2I Adapter',
  t5_encoder: 'T5 encoder',
  text_llm: 'text LLM',
  unknown: 'model',
  vae: 'VAE',
  wan_t5_encoder: 'Wan T5 encoder',
};

const humanize = (value: string): string => value.replaceAll(/[_-]+/g, ' ');

const getBaseLabel = (base: string): string => BASE_LABELS[base] ?? humanize(base);

const getModelTypeLabel = (modelType: string): string => MODEL_TYPE_LABELS[modelType] ?? humanize(modelType);

const capitalize = (value: string): string => (value.length === 0 ? value : value[0]?.toUpperCase() + value.slice(1));

const buildSlotLabel = (fieldTemplate: FieldInputTemplate, base: string | null, modelType: string | null): string => {
  const baseLabel = base ? getBaseLabel(base) : null;
  const typeLabel = modelType ? getModelTypeLabel(modelType) : null;

  if (baseLabel && typeLabel) {
    return `${baseLabel} ${typeLabel}`;
  }

  if (baseLabel) {
    return `${baseLabel} model`;
  }

  if (typeLabel) {
    return capitalize(typeLabel);
  }

  return fieldTemplate.title || fieldTemplate.name;
};

interface DuckTypedModelValue {
  key: string;
  hash?: unknown;
  name?: unknown;
  base?: unknown;
  type?: unknown;
}

const isDuckTypedModelValue = (value: unknown): value is DuckTypedModelValue =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as Record<string, unknown>).key === 'string' &&
  ((value as Record<string, unknown>).key as string).trim().length > 0;

const buildIdentifier = (value: DuckTypedModelValue): WorkflowModelIdentifier => {
  const identifier: WorkflowModelIdentifier = { key: value.key };

  if (typeof value.hash === 'string') {
    identifier.hash = value.hash;
  }

  if (typeof value.name === 'string') {
    identifier.name = value.name;
  }

  if (typeof value.base === 'string') {
    identifier.base = value.base;
  }

  if (typeof value.type === 'string') {
    identifier.type = value.type;
  }

  return identifier;
};

const mostFrequentBase = (counts: Map<string, number>): string | null => {
  let best: string | null = null;
  let bestCount = 0;

  for (const [base, count] of counts) {
    if (count > bestCount) {
      best = base;
      bestCount = count;
    }
  }

  return best;
};

const computePrimaryBase = (requirements: readonly WorkflowModelRequirement[]): string | null => {
  const mainBaseCounts = new Map<string, number>();
  const allBaseCounts = new Map<string, number>();

  const bump = (counts: Map<string, number>, base: string): void => {
    counts.set(base, (counts.get(base) ?? 0) + 1);
  };

  for (const requirement of requirements) {
    if (requirement.kind === 'exact') {
      if (requirement.identifier.base) {
        bump(allBaseCounts, requirement.identifier.base);

        if (requirement.identifier.type === 'main') {
          bump(mainBaseCounts, requirement.identifier.base);
        }
      }
    } else if (requirement.base) {
      bump(allBaseCounts, requirement.base);
    }
  }

  return mostFrequentBase(mainBaseCounts) ?? mostFrequentBase(allBaseCounts);
};

/**
 * Classify concrete model identifiers as exact requirements and empty required inputs as slots; skip
 * connection-fed inputs owned upstream.
 */
export const extractWorkflowModelRequirements = (
  document: ProjectGraphState,
  templates: InvocationTemplates
): WorkflowModelRequirementSet => {
  const connectionTargets = new Set(document.edges.map((edge) => `${edge.target}::${edge.targetHandle}`));

  const requirements: WorkflowModelRequirement[] = [];
  const exactKeyHashes = new Map<string, string | undefined>();
  const seenExactDedupeKeys = new Set<string>();
  const seenSlotSignatures = new Set<string>();

  for (const node of document.nodes) {
    if (!isInvocationNode(node)) {
      continue;
    }

    const template = templates[node.data.type];

    if (!template) {
      continue;
    }

    const addExactRequirement = (value: unknown): void => {
      if (!isDuckTypedModelValue(value)) {
        return;
      }

      const identifier = buildIdentifier(value);

      let dedupeKey = identifier.key;

      if (exactKeyHashes.has(identifier.key)) {
        const existingHash = exactKeyHashes.get(identifier.key);

        if (identifier.hash !== undefined && existingHash !== undefined && existingHash !== identifier.hash) {
          dedupeKey = `${identifier.key}::${identifier.hash}`;
        }
      } else {
        exactKeyHashes.set(identifier.key, identifier.hash);
      }

      if (seenExactDedupeKeys.has(dedupeKey)) {
        return;
      }

      seenExactDedupeKeys.add(dedupeKey);
      requirements.push({ identifier, kind: 'exact', label: identifier.name ?? identifier.key });
    };

    for (const fieldTemplate of Object.values({ ...template.inputs, ...node.data.dynamicInputTemplates })) {
      const isLoraCollection = fieldTemplate.type.name === 'LoRAField';

      if (!isModelFieldType(fieldTemplate.type) && !isLoraCollection) {
        continue;
      }

      if (connectionTargets.has(`${node.id}::${fieldTemplate.name}`)) {
        continue;
      }

      const value = node.data.inputs[fieldTemplate.name]?.value;

      // Inspect nested identifiers in LoRA collections so their requirements match equivalent selector-node
      // graphs.
      if (isLoraCollection) {
        const entries = toLoraFieldCollectionList(value);

        for (const entry of entries) {
          if (isLoraFieldCollectionEntry(entry)) {
            addExactRequirement(entry.lora);
          }
        }

        // An empty collection still falls through to the slot path below, so a required-but-blank
        // field keeps describing what it needs the way every other blank model field does.
        if (entries.length > 0) {
          continue;
        }
      }

      if (isDuckTypedModelValue(value)) {
        addExactRequirement(value);
        continue;
      }

      if (!fieldTemplate.required) {
        continue;
      }

      const base = fieldTemplate.uiModelBase?.[0] ?? null;
      const modelType = fieldTemplate.uiModelType?.[0] ?? null;
      const signature = `${base ?? ''}::${modelType ?? ''}`;

      if (seenSlotSignatures.has(signature)) {
        continue;
      }

      seenSlotSignatures.add(signature);
      requirements.push({ base, kind: 'slot', label: buildSlotLabel(fieldTemplate, base, modelType), modelType });
    }
  }

  return { primaryBase: computePrimaryBase(requirements), requirements };
};

export interface InstalledModelSummary {
  key: string;
  hash: string;
  name: string;
  base: string;
  type: string;
}

export interface StarterCatalogEntry {
  name: string;
  previous_names?: string[] | null;
  base: string;
  type: string;
  source: string;
  is_installed: boolean;
  dependencies?: readonly { source: string; is_installed: boolean }[] | null;
}

export type ModelRequirementStatus = 'installed' | 'installing' | 'installable' | 'unresolvable';

export interface ResolvedModelRequirement {
  requirement: WorkflowModelRequirement;
  status: ModelRequirementStatus;
  matchedModelName: string | null;
  starterMatch: StarterCatalogEntry | null;
}

interface ResolutionDeps {
  installedModels: readonly InstalledModelSummary[];
  starterModels: readonly StarterCatalogEntry[];
  activeInstallSources: ReadonlySet<string>;
}

const findInstalledForExact = (
  identifier: WorkflowModelIdentifier,
  installedModels: readonly InstalledModelSummary[]
): InstalledModelSummary | null => {
  const byKey = installedModels.find((model) => model.key === identifier.key);

  if (byKey) {
    return byKey;
  }

  if (identifier.hash) {
    const byHash = installedModels.find((model) => model.hash === identifier.hash);

    if (byHash) {
      return byHash;
    }
  }

  if (identifier.name && identifier.base && identifier.type) {
    const byNameBaseType = installedModels.find(
      (model) => model.name === identifier.name && model.base === identifier.base && model.type === identifier.type
    );

    if (byNameBaseType) {
      return byNameBaseType;
    }
  }

  return null;
};

const findInstalledForSlot = (
  base: string | null,
  modelType: string | null,
  installedModels: readonly InstalledModelSummary[]
): InstalledModelSummary | null =>
  installedModels.find(
    (model) => (base === null || model.base === base) && (modelType === null || model.type === modelType)
  ) ?? null;

const matchesStarterName = (starter: StarterCatalogEntry, name: string): boolean => {
  const target = name.toLowerCase();

  if (starter.name.toLowerCase() === target) {
    return true;
  }

  return (starter.previous_names ?? []).some((previousName) => previousName.toLowerCase() === target);
};

const findStarterForExact = (
  identifier: WorkflowModelIdentifier,
  starterModels: readonly StarterCatalogEntry[]
): StarterCatalogEntry | null => {
  if (!identifier.name || !identifier.base || !identifier.type) {
    return null;
  }

  return (
    starterModels.find(
      (starter) =>
        matchesStarterName(starter, identifier.name as string) &&
        starter.base === identifier.base &&
        starter.type === identifier.type
    ) ?? null
  );
};

const findStarterForSlot = (
  base: string | null,
  modelType: string | null,
  starterModels: readonly StarterCatalogEntry[]
): StarterCatalogEntry | null =>
  starterModels.find(
    (starter) => (base === null || starter.base === base) && (modelType === null || starter.type === modelType)
  ) ?? null;

const isStarterActivelyInstalling = (
  starter: StarterCatalogEntry,
  activeInstallSources: ReadonlySet<string>
): boolean =>
  activeInstallSources.has(starter.source) ||
  (starter.dependencies ?? []).some((dependency) => activeInstallSources.has(dependency.source));

const resolveFromStarter = (
  requirement: WorkflowModelRequirement,
  starter: StarterCatalogEntry | null,
  activeInstallSources: ReadonlySet<string>
): ResolvedModelRequirement => {
  if (!starter) {
    return { matchedModelName: null, requirement, starterMatch: null, status: 'unresolvable' };
  }

  return {
    matchedModelName: null,
    requirement,
    starterMatch: starter,
    status: isStarterActivelyInstalling(starter, activeInstallSources) ? 'installing' : 'installable',
  };
};

/**
 * Resolve exact requirements by key, hash, then name/base/type; slots by base/type. Starter fallback reports
 * installing sources, including dependencies.
 */
export const resolveWorkflowModelRequirements = (
  requirements: readonly WorkflowModelRequirement[],
  deps: ResolutionDeps
): ResolvedModelRequirement[] =>
  requirements.map((requirement) => {
    if (requirement.kind === 'exact') {
      const installed = findInstalledForExact(requirement.identifier, deps.installedModels);

      if (installed) {
        return { matchedModelName: installed.name, requirement, starterMatch: null, status: 'installed' };
      }

      return resolveFromStarter(
        requirement,
        findStarterForExact(requirement.identifier, deps.starterModels),
        deps.activeInstallSources
      );
    }

    const installed = findInstalledForSlot(requirement.base, requirement.modelType, deps.installedModels);

    if (installed) {
      return { matchedModelName: installed.name, requirement, starterMatch: null, status: 'installed' };
    }

    return resolveFromStarter(
      requirement,
      findStarterForSlot(requirement.base, requirement.modelType, deps.starterModels),
      deps.activeInstallSources
    );
  });

/**
 * Link only missing requirements to Add Models. Prefer catalog names, then exact model names or raw slot
 * base/type; display prose is not searchable taxonomy.
 */
export const getAddModelsSearchTerm = (resolved: ResolvedModelRequirement): string | null => {
  if (resolved.status === 'installed') {
    return null;
  }

  const starterName = resolved.starterMatch?.name.trim();

  if (starterName) {
    return starterName;
  }

  const { requirement } = resolved;

  if (requirement.kind === 'exact') {
    return requirement.identifier.name?.trim() || requirement.identifier.base?.trim() || null;
  }

  return requirement.base?.trim() || requirement.modelType?.trim() || null;
};

// #endregion
