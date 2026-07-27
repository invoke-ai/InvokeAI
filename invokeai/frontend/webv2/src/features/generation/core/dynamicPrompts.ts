/**
 * Dynamic prompting policy: detecting the syntax, bounding the expansion, and
 * turning an expanded prompt list into the batch matrix the backend expects.
 *
 * The backend expands `a {red|green} {cat|dog}` into concrete prompts; the
 * frontend never rebuilds the graph per prompt. Instead the expansion becomes a
 * batch dimension over the graph's existing `positive_prompt` string node, which
 * is why every decision here is expressed as `BatchDatum` groups rather than as
 * a list of graphs.
 *
 * Batch semantics (see `invokeai/app/services/session_queue/session_queue_common.py`):
 * the outer list is a cartesian PRODUCT of groups, and each inner group is
 * ZIPPED (all its items must have the same length).
 */

import { sanitizeBatchCount } from './batch';

// Local copy: `core/settings.ts` exports the same bound but imports this module,
// so taking it from there would be a cycle.
const SEED_MAX = 4_294_967_295;

/** Matches the backend's `max_prompts: int = Body(ge=1, le=10000)`. */
export const DYNAMIC_PROMPTS_MIN_PROMPTS = 1;
export const DYNAMIC_PROMPTS_MAX_PROMPTS = 10_000;
export const DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS = 100;

export type DynamicPromptsSeedBehaviour = 'per-iteration' | 'per-image';

export const DYNAMIC_PROMPTS_SEED_BEHAVIOURS: readonly DynamicPromptsSeedBehaviour[] = ['per-iteration', 'per-image'];

export interface DynamicPromptsConfig {
  combinatorial: boolean;
  maxPrompts: number;
  seedBehaviour: DynamicPromptsSeedBehaviour;
}

/**
 * Whether a prompt is worth sending to the expansion route at all. A prompt with
 * no `{…}` is its own single expansion, so the round trip is skipped entirely.
 *
 * Deliberately does not look for `__wildcard__`: this deployment runs an
 * unconfigured `WildcardManager`, so wildcards never resolve.
 */
export const hasDynamicPromptSyntax = (prompt: string): boolean => /\{[\s\S]*\}/.test(prompt);

export const isDynamicPromptsSeedBehaviour = (value: unknown): value is DynamicPromptsSeedBehaviour =>
  value === 'per-iteration' || value === 'per-image';

export const sanitizeMaxPrompts = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value)
    ? Math.min(DYNAMIC_PROMPTS_MAX_PROMPTS, Math.max(DYNAMIC_PROMPTS_MIN_PROMPTS, Math.round(value)))
    : DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS;

/** Reads an untrusted persisted/transported config, or `null` when it is unusable. */
export const sanitizeDynamicPromptsConfig = (value: unknown): DynamicPromptsConfig | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const config = value as Partial<DynamicPromptsConfig>;

  return {
    combinatorial: config.combinatorial !== false,
    maxPrompts: sanitizeMaxPrompts(config.maxPrompts),
    seedBehaviour: isDynamicPromptsSeedBehaviour(config.seedBehaviour) ? config.seedBehaviour : 'per-iteration',
  };
};

export const generateSeedSequence = (start: number, count: number): number[] =>
  Array.from({ length: Math.max(1, Math.round(count)) }, (_, index) => (start + index) % SEED_MAX);

export interface GeneratePromptBatchDatum {
  field_name: 'value';
  items: (number | string)[];
  node_path: string;
}

export interface GeneratePromptBatchPlanInput {
  batchCount: number;
  negativePrompt: string;
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  prompts: readonly string[];
  seed: number;
  seedBehaviour: DynamicPromptsSeedBehaviour;
  seedNodeId: string;
  shouldRandomizeSeed: boolean;
}

export interface GeneratePromptBatchPlan {
  /** Outer list is a cartesian product; each inner list is zipped. */
  data: GeneratePromptBatchDatum[][];
  runs: number;
  /** Images this plan produces, for optimistic placeholder sizing. */
  expectedImageCount: number;
}

/**
 * Builds the `batch.data` / `batch.runs` payload for a generate submission.
 *
 * With a single prompt this reproduces the pre-dynamic-prompts payload exactly,
 * which `dynamicPrompts.test.ts` pins:
 * - randomized seed -> one zipped group of `batchCount` seeds and repeated
 *   prompts, `runs: 1`
 * - fixed seed -> one zipped group of length 1, `runs: batchCount`
 *
 * With several prompts the seed behaviour decides the shape:
 * - `per-iteration` -> seeds become their own group, so the product is
 *   `iterations x prompts` and every prompt in an iteration shares its seed
 * - `per-image` -> one distinct sequential seed per image, zipped against the
 *   prompt list repeated `batchCount` times, `runs: 1`
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
  shouldRandomizeSeed,
}: GeneratePromptBatchPlanInput): GeneratePromptBatchPlan => {
  const iterations = sanitizeBatchCount(batchCount);
  const promptList = prompts.length > 0 ? [...prompts] : [''];
  const promptDatum = (items: string[]): GeneratePromptBatchDatum[] => [
    { field_name: 'value', items, node_path: positivePromptNodeId },
    { field_name: 'value', items: items.map(() => negativePrompt), node_path: negativePromptNodeId },
  ];

  // A single prompt keeps the legacy shape: seeds and prompts stay in one zipped
  // group so the payload is byte-identical to what shipped before this feature.
  if (promptList.length === 1) {
    const seeds = shouldRandomizeSeed ? generateSeedSequence(seed, iterations) : [seed];

    return {
      data: [
        [{ field_name: 'value', items: seeds, node_path: seedNodeId }, ...promptDatum(seeds.map(() => promptList[0]))],
      ],
      expectedImageCount: iterations,
      runs: shouldRandomizeSeed ? 1 : iterations,
    };
  }

  if (seedBehaviour === 'per-image') {
    const seeds = generateSeedSequence(seed, promptList.length * iterations);
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
  const seeds = shouldRandomizeSeed ? generateSeedSequence(seed, iterations) : [seed];

  return {
    data: [[{ field_name: 'value', items: seeds, node_path: seedNodeId }], promptDatum(promptList)],
    expectedImageCount: promptList.length * iterations,
    runs: shouldRandomizeSeed ? 1 : iterations,
  };
};
