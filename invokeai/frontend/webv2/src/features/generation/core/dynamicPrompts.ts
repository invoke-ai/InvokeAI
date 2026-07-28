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
 * One `/`-separated segment of a wildcard name. Mirrors the backend's
 * `_WILDCARD_NAME_SEGMENT`: it must begin and end alphanumerically, so
 * `snake__case` is ordinary text rather than a reference.
 */
const WILDCARD_NAME_SEGMENT = '[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?';
const WILDCARD_NAME_SOURCE = `${WILDCARD_NAME_SEGMENT}(?:/${WILDCARD_NAME_SEGMENT})*`;

/**
 * A referenced path, which is looser than a storable name: `*` may stand
 * anywhere in it, and it may cross `/` while doing so. `__animals/*__`,
 * `__colour*__` and a bare `__*__` are all things the backend resolves.
 */
const WILDCARD_PATH_SEGMENT = '[A-Za-z0-9*](?:[A-Za-z0-9_*-]*[A-Za-z0-9*])?';
const WILDCARD_PATH_SOURCE = `${WILDCARD_PATH_SEGMENT}(?:/${WILDCARD_PATH_SEGMENT})*`;

/**
 * `__name__`, the reference form for a wildcard. Shared with the syntax scanner
 * so highlighting and expansion agree on what counts as a wildcard.
 *
 * Three optional parts ride along, all of which the backend accepts and none of
 * which is part of the name being looked up:
 *
 * - a sampler override, `__~name__` (random) or `__@name__` (cyclical);
 * - a glob, `__artists/*__`, resolving to every matching wildcard at once;
 * - parameters, `__name(mood=warm)__`, binding `${mood}` inside the values.
 *
 * Capture group 1 is the path alone — the part to look up.
 */
export const WILDCARD_REFERENCE_RE = new RegExp(`__[~@]?(${WILDCARD_PATH_SOURCE})(?:\\([^()]*\\))?__`);

/** The same rule anchored, for validating a name the user is typing. */
export const WILDCARD_NAME_RE = new RegExp(`^${WILDCARD_NAME_SOURCE}$`);

/**
 * Whether `name` matches a glob, given the literal runs between its `*`s.
 *
 * Deliberately not a regex. Translating `a*b*c` to `^a.*b.*c$` reads well but
 * makes adjacent stars into `.*.*`, which backtracks exponentially: a dozen of
 * them against one catalog name took a minute to fail. The path comes straight
 * out of whatever the user is typing, so that is a frozen tab, not a slow test.
 *
 * The greedy scan below is the standard `*`-only matcher and is linear in
 * practice: anchor the first run to the start and the last to the end, then take
 * each middle run at its earliest remaining occurrence. Earliest is always safe
 * when the only wildcard is "any run of anything" — consuming less can never
 * turn a match into a miss.
 */
const matchesGlobLiterals = (literals: readonly string[], name: string): boolean => {
  const first = literals[0] ?? '';
  const last = literals[literals.length - 1] ?? '';

  if (literals.length === 1) {
    return name === first;
  }

  // Checked before the middle runs so the two anchors cannot overlap and count
  // the same characters twice: `ab*ba` needs four characters, not three.
  if (name.length < first.length + last.length || !name.startsWith(first) || !name.endsWith(last)) {
    return false;
  }

  const end = name.length - last.length;
  let index = first.length;

  for (let position = 1; position < literals.length - 1; position++) {
    const literal = literals[position];

    if (literal === undefined || literal.length === 0) {
      continue;
    }

    const found = name.indexOf(literal, index);

    if (found < 0 || found + literal.length > end) {
      return false;
    }

    index = found + literal.length;
  }

  return true;
};

/**
 * Whether a referenced path resolves against the catalog.
 *
 * A glob matches when *any* known name matches it, mirroring the backend, where
 * `*` is an unanchored run of any characters — `/` included, so `__*s__` reaches
 * `animals/dogs`. `**` is spelt differently but means the same thing there, and
 * falls out of the same translation.
 *
 * A path matching nothing is still worth flagging: the expansion leaves the
 * literal `__name__` in the prompt rather than failing, so an unnoticed typo
 * goes to the model as text.
 */
export const matchesKnownWildcard = (path: string, knownNames: ReadonlySet<string>): boolean => {
  if (!path.includes('*')) {
    return knownNames.has(path);
  }

  const literals = path.split('*');

  for (const name of knownNames) {
    if (matchesGlobLiterals(literals, name)) {
      return true;
    }
  }

  return false;
};

/** Mirrors the backend's `MAX_WILDCARD_NAME_LENGTH`. */
export const MAX_WILDCARD_NAME_LENGTH = 128;

/**
 * The backend's bounds on a values list, mirrored here so both ways of writing
 * one can say what is wrong before sending it.
 *
 * These lived next to the import planner, which meant an import pre-flighted
 * them politely while the editor discovered them as a raw error from the server.
 */
export const MAX_WILDCARD_VALUES = 10_000;
export const MAX_WILDCARD_VALUE_LENGTH = 2_000;

/** Why a values list would be rejected, or `null` if it would be accepted. */
export const getWildcardValuesError = (values: readonly string[]): 'tooManyValues' | 'valueTooLong' | null => {
  if (values.length > MAX_WILDCARD_VALUES) {
    return 'tooManyValues';
  }
  // Kept apart from the count: told "too many values", someone holding three
  // values and one long paragraph has no idea which one to shorten.
  if (values.some((value) => value.length > MAX_WILDCARD_VALUE_LENGTH)) {
    return 'valueTooLong';
  }

  return null;
};

/**
 * The values as they will be stored: comments removed, trimmed, blanks dropped.
 *
 * Applied wherever a list is authored, so that what is saved is what an export
 * writes and a re-import reads back. Without it a stray blank line survived into
 * the catalog and vanished on the next round trip.
 *
 * `#` comments to end of line, and the expander applies that to a substituted
 * value as readily as to the prompt around it — a value of `poster #1` reaches
 * the model as `poster `, and `#ff0000 glow` reaches it as nothing at all. There
 * is no escape for it upstream. Dropping the comment here is what makes that
 * visible while the list is in front of the user, rather than at generation time
 * with no clue as to where the text went. The `.txt` reader has always done
 * this, being the only format with no parser of its own; the rule belongs to the
 * value rather than to the file it arrived in.
 */
export const normalizeWildcardValues = (values: readonly string[]): string[] =>
  values.map((value) => value.split('#')[0]?.trim() ?? '').filter((value) => value.length > 0);

/**
 * Why a wildcard name would be rejected, or `null` if it would be accepted.
 *
 * The server remains the authority — only it can see the whole catalog at save
 * time — but a name is typed character by character, and finding out it was
 * invalid only after a round trip is a poor way to learn the rule.
 */
export const getWildcardNameError = (
  name: string,
  existingNames: ReadonlySet<string> = new Set()
): 'empty' | 'invalid' | 'taken' | 'tooLong' | null => {
  const normalized = name.trim();

  if (!normalized) {
    return 'empty';
  }
  if (normalized.length > MAX_WILDCARD_NAME_LENGTH) {
    return 'tooLong';
  }
  if (!WILDCARD_NAME_RE.test(normalized)) {
    return 'invalid';
  }
  if (existingNames.has(normalized)) {
    return 'taken';
  }

  return null;
};

/**
 * Whether a prompt is worth sending to the expansion route at all. A prompt with
 * no dynamic syntax in it is its own single expansion, so the round trip is
 * skipped entirely.
 *
 * `#` counts, because upstream treats it as a comment to end of line and strips
 * it — with no way to escape it. Leaving it out would make whether a `#` reaches
 * the model depend on whether the prompt happened to contain a `{…}` elsewhere,
 * which is a worse surprise than the round trip.
 */
export const hasDynamicPromptSyntax = (prompt: string): boolean =>
  /\{[\s\S]*\}/.test(prompt) || prompt.includes('#') || WILDCARD_REFERENCE_RE.test(prompt);

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
