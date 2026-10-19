/**
 * Prompts arrive pre-expanded. Backend outer groups form a Cartesian product; entries within each group zip and
 * require equal lengths.
 */

import { SEED_MAX } from '@platform/core/seed';

export const MAX_QUEUE_BATCH_ITEMS = 10_000;

export type QueuePromptSeedBehaviour = 'per-iteration' | 'per-image';

/** Direction between consecutive seeds of one submission; 0 holds the start seed for every image. */
export type QueueSeedStep = -1 | 0 | 1;

export const isQueuePromptSeedBehaviour = (value: unknown): value is QueuePromptSeedBehaviour =>
  value === 'per-iteration' || value === 'per-image';

export const isQueueSeedStep = (value: unknown): value is QueueSeedStep => value === -1 || value === 0 || value === 1;

export const sanitizeBatchCount = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value)
    ? Math.min(MAX_QUEUE_BATCH_ITEMS, Math.max(1, Math.round(value)))
    : 1;

/** Consecutive seeds from `start`, wrapping over the inclusive `0..SEED_MAX` range. */
export const generateSeedSequence = (start: number, count: number, step: QueueSeedStep = 1): number[] => {
  const range = SEED_MAX + 1;

  return Array.from(
    { length: sanitizeBatchCount(count) },
    (_, index) => (((start + index * step) % range) + range) % range
  );
};

/** A value the backend substitutes into a node field per session. */
export type QueueBatchItem = number | string | { image_name: string };

/** One value list of a zipped batch group, in the backend's `BatchDatum` shape. */
export interface QueueBatchDatum {
  field_name: string;
  items: QueueBatchItem[];
  node_path: string;
}

/** A batch-node value list, as a workflow submission persists it (camelCase, like `QueueWorkflowSeed`). */
export interface QueueWorkflowBatchDatum {
  fieldName: string;
  items: QueueBatchItem[];
  nodeId: string;
}

const isQueueBatchItem = (value: unknown): value is QueueBatchItem =>
  typeof value === 'string' ||
  (typeof value === 'number' && Number.isFinite(value)) ||
  (typeof value === 'object' &&
    value !== null &&
    typeof (value as { image_name?: unknown }).image_name === 'string' &&
    (value as { image_name: string }).image_name.length > 0);

export const isQueueWorkflowBatchDatum = (value: unknown): value is QueueWorkflowBatchDatum => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const datum = value as Partial<QueueWorkflowBatchDatum>;

  return (
    typeof datum.fieldName === 'string' &&
    datum.fieldName.length > 0 &&
    typeof datum.nodeId === 'string' &&
    datum.nodeId.length > 0 &&
    Array.isArray(datum.items) &&
    datum.items.length > 0 &&
    datum.items.length <= MAX_QUEUE_BATCH_ITEMS &&
    datum.items.every(isQueueBatchItem)
  );
};

export interface GeneratePromptBatchDatum extends QueueBatchDatum {
  field_name: 'value';
}

/** Persist each workflow seed's start and step; expand deterministic runs at send time. */
export interface QueueWorkflowSeed {
  fieldName: string;
  nodeId: string;
  seed: number;
  seedStep: -1 | 1;
}

export const isQueueWorkflowSeed = (value: unknown): value is QueueWorkflowSeed => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const seed = value as Partial<QueueWorkflowSeed>;

  return (
    typeof seed.fieldName === 'string' &&
    seed.fieldName.length > 0 &&
    typeof seed.nodeId === 'string' &&
    seed.nodeId.length > 0 &&
    typeof seed.seed === 'number' &&
    Number.isInteger(seed.seed) &&
    seed.seed >= 0 &&
    seed.seed <= SEED_MAX &&
    (seed.seedStep === -1 || seed.seedStep === 1)
  );
};

export interface WorkflowBatchPlan {
  /** Batch groups in the backend's shape, or undefined when one graph repeats unchanged. */
  data?: QueueBatchDatum[][];
  runs: number;
}

const toBackendDatum = (datum: QueueWorkflowBatchDatum): QueueBatchDatum => ({
  field_name: datum.fieldName,
  items: datum.items,
  node_path: datum.nodeId,
});

/**
 * Every combination the backend would produce from `groups`, in its order: the first group varies slowest, and
 * each group's datums are read in step. Each entry maps a datum to the index of the item that combination uses.
 */
const expandCombinations = (groups: readonly (readonly QueueWorkflowBatchDatum[])[]): number[][] => {
  let combinations: number[][] = [[]];

  for (const group of groups) {
    const length = group[0]?.items.length ?? 0;
    const next: number[][] = [];

    for (const prefix of combinations) {
      for (let index = 0; index < length; index += 1) {
        next.push([...prefix, index]);
      }
    }

    combinations = next;
  }

  return combinations;
};

/**
 * Without varying seeds the backend multiplies the batch groups itself and repeats the product `batchCount` times.
 * Once a seed steps, every session must get its own seed, so the product is expanded here into one zipped group,
 * run-major, with each seed walking straight through it.
 */
export const buildQueueWorkflowBatchPlan = ({
  batchCount,
  batchData,
  seeds,
}: {
  batchCount: number;
  batchData: readonly (readonly QueueWorkflowBatchDatum[])[] | undefined;
  seeds: readonly QueueWorkflowSeed[] | undefined;
}): WorkflowBatchPlan => {
  const runs = sanitizeBatchCount(batchCount);
  const groups = (batchData ?? []).filter((group) => group.length > 0);
  const hasSeeds = !!seeds && seeds.length > 0;

  if (!hasSeeds || (runs === 1 && groups.length === 0)) {
    return groups.length > 0 ? { data: groups.map((group) => group.map(toBackendDatum)), runs } : { runs };
  }

  if (groups.length === 0) {
    return {
      data: [
        seeds.map((seed) => ({
          field_name: seed.fieldName,
          items: generateSeedSequence(seed.seed, runs, seed.seedStep),
          node_path: seed.nodeId,
        })),
      ],
      runs: 1,
    };
  }

  const combinations = expandCombinations(groups);
  const total = combinations.length * runs;
  const zipped: QueueBatchDatum[] = seeds.map((seed) => ({
    field_name: seed.fieldName,
    items: generateSeedSequence(seed.seed, total, seed.seedStep),
    node_path: seed.nodeId,
  }));

  groups.forEach((group, groupIndex) => {
    for (const datum of group) {
      const items: QueueBatchItem[] = [];

      for (let run = 0; run < runs; run += 1) {
        for (const combination of combinations) {
          items.push(datum.items[combination[groupIndex] as number] as QueueBatchItem);
        }
      }

      zipped.push({ field_name: datum.fieldName, items, node_path: datum.nodeId });
    }
  });

  return { data: [zipped], runs: 1 };
};

export interface GeneratePromptBatchPlanInput {
  batchCount: number;
  negativePrompt: string;
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  prompts: readonly string[];
  seed: number;
  seedBehaviour: QueuePromptSeedBehaviour;
  seedNodeId: string;
  seedStep: QueueSeedStep;
}

export interface GeneratePromptBatchPlan {
  /** Outer list is a cartesian product; each inner list is zipped. */
  data: GeneratePromptBatchDatum[][];
  runs: number;
  /** Images this plan produces, for optimistic placeholder sizing. */
  expectedImageCount: number;
}

/** The pre-seed-mode sequence: consecutive from `start`, wrapping one short of the inclusive range. */
const generateLegacySeedSequence = (start: number, count: number): number[] =>
  Array.from({ length: sanitizeBatchCount(count) }, (_, index) => (start + index) % SEED_MAX);

/**
 * Recovery preserves legacy random-toggle seed rules, including exclusive SEED_MAX wrapping; new submissions never
 * use this path.
 */
export const buildLegacyGeneratePromptBatchPlan = ({
  batchCount,
  negativePrompt,
  negativePromptNodeId,
  positivePromptNodeId,
  prompts,
  seed,
  seedBehaviour,
  seedNodeId,
  seedStep,
}: GeneratePromptBatchPlanInput): GeneratePromptBatchPlan => {
  const shouldRandomizeSeed = seedStep !== 0;
  const iterations = sanitizeBatchCount(batchCount);
  const promptList = prompts.length > 0 ? [...prompts] : [''];
  const promptDatum = (items: string[]): GeneratePromptBatchDatum[] => [
    { field_name: 'value', items, node_path: positivePromptNodeId },
    { field_name: 'value', items: items.map(() => negativePrompt), node_path: negativePromptNodeId },
  ];

  if (promptList.length === 1) {
    const seeds = shouldRandomizeSeed ? generateLegacySeedSequence(seed, iterations) : [seed];

    return {
      data: [
        [{ field_name: 'value', items: seeds, node_path: seedNodeId }, ...promptDatum(seeds.map(() => promptList[0]))],
      ],
      expectedImageCount: iterations,
      runs: shouldRandomizeSeed ? 1 : iterations,
    };
  }

  if (seedBehaviour === 'per-image') {
    const seeds = generateLegacySeedSequence(seed, promptList.length * iterations);
    const repeatedPrompts = Array.from({ length: iterations }, () => promptList).flat();

    return {
      data: [[{ field_name: 'value', items: seeds, node_path: seedNodeId }, ...promptDatum(repeatedPrompts)]],
      expectedImageCount: seeds.length,
      runs: 1,
    };
  }

  const seeds = shouldRandomizeSeed ? generateLegacySeedSequence(seed, iterations) : [seed];

  return {
    data: [[{ field_name: 'value', items: seeds, node_path: seedNodeId }], promptDatum(promptList)],
    expectedImageCount: promptList.length * iterations,
    runs: shouldRandomizeSeed ? 1 : iterations,
  };
};

/**
 * Held seeds reuse one value. Stepping per-iteration seeds form a product with prompts; per-image seeds zip with
 * repeated prompts.
 */
export const buildGeneratePromptBatchPlan = ({
  batchCount,
  negativePrompt,
  negativePromptNodeId,
  positivePromptNodeId,
  prompts,
  seed,
  seedBehaviour,
  seedNodeId,
  seedStep,
}: GeneratePromptBatchPlanInput): GeneratePromptBatchPlan => {
  const iterations = sanitizeBatchCount(batchCount);
  const promptList = prompts.length > 0 ? [...prompts] : [''];
  const promptDatum = (items: string[]): GeneratePromptBatchDatum[] => [
    { field_name: 'value', items, node_path: positivePromptNodeId },
    { field_name: 'value', items: items.map(() => negativePrompt), node_path: negativePromptNodeId },
  ];
  const isHeld = seedStep === 0;

  if (promptList.length === 1) {
    const seeds = isHeld ? [seed] : generateSeedSequence(seed, iterations, seedStep);

    return {
      data: [
        [{ field_name: 'value', items: seeds, node_path: seedNodeId }, ...promptDatum(seeds.map(() => promptList[0]))],
      ],
      expectedImageCount: iterations,
      runs: isHeld ? iterations : 1,
    };
  }

  if (seedBehaviour === 'per-image' && !isHeld) {
    const seeds = generateSeedSequence(seed, promptList.length * iterations, seedStep);
    const repeatedPrompts = Array.from({ length: iterations }, () => promptList).flat();

    return {
      data: [[{ field_name: 'value', items: seeds, node_path: seedNodeId }, ...promptDatum(repeatedPrompts)]],
      expectedImageCount: seeds.length,
      runs: 1,
    };
  }

  // per-iteration: the seed list is its own dimension, so each iteration's seed
  // is applied across the whole prompt set. Seeds first so results group by
  // iteration rather than interleaving prompts.
  const seeds = isHeld ? [seed] : generateSeedSequence(seed, iterations, seedStep);

  return {
    data: [[{ field_name: 'value', items: seeds, node_path: seedNodeId }], promptDatum(promptList)],
    expectedImageCount: promptList.length * iterations,
    runs: isHeld ? iterations : 1,
  };
};
