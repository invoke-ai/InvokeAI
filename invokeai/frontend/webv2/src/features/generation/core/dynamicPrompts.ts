/**
 * Dynamic prompting policy: detecting the syntax and bounding the expansion.
 *
 * The backend expands `a {red|green} {cat|dog}` into concrete prompts; the
 * frontend never rebuilds the graph per prompt. Turning the expanded list into a
 * batch dimension over the graph's existing `positive_prompt` string node is
 * Queue's job — see `features/queue/core/promptBatch.ts`.
 */

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
  /**
   * Seeds the random sampler. Held apart from the generation seed and stable
   * until the user shuffles, so the previewed prompts are the prompts that
   * generate. Unused when combinatorial.
   */
  sampleSeed: number;
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
    sampleSeed: sanitizeSampleSeed(config.sampleSeed),
    seedBehaviour: isDynamicPromptsSeedBehaviour(config.seedBehaviour) ? config.seedBehaviour : 'per-iteration',
  };
};

export const sanitizeSampleSeed = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.abs(Math.round(value)) % SEED_MAX : 0;

export const createDynamicPromptsSampleSeed = (): number => Math.floor(Math.random() * SEED_MAX);
