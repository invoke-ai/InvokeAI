/**
 * The seed/prompt matrix for a generate submission.
 *
 * A generate graph carries one `positive_prompt` string node, so submitting
 * several prompts is a batch dimension over that node rather than several
 * graphs. Backend batch semantics (see
 * `invokeai/app/services/session_queue/session_queue_common.py`): the outer list
 * is a cartesian PRODUCT of groups, and each inner group is ZIPPED, so all of
 * its items must have the same length.
 *
 * The prompt list arrives already expanded — Queue never talks to the expansion
 * route itself.
 */

const SEED_MAX = 4_294_967_295;

export type QueuePromptSeedBehaviour = 'per-iteration' | 'per-image';

export const isQueuePromptSeedBehaviour = (value: unknown): value is QueuePromptSeedBehaviour =>
  value === 'per-iteration' || value === 'per-image';

export const sanitizeBatchCount = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.max(1, Math.round(value)) : 1;

export const generateSeedSequence = (start: number, count: number): number[] =>
  Array.from({ length: sanitizeBatchCount(count) }, (_, index) => (start + index) % SEED_MAX);

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
  seedBehaviour: QueuePromptSeedBehaviour;
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
 * With a single prompt this reproduces the pre-dynamic-prompts payload exactly,
 * which `promptBatch.test.ts` pins:
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
