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

/** The same rule anchored, for validating a name the user is typing. */
export const WILDCARD_NAME_RE = new RegExp(`^${WILDCARD_NAME_SOURCE}$`);

export interface WildcardReference {
  /** The path sent to the wildcard manager, without sampler or parameters. */
  lookupPath: string;
  /** Full `__…__` span in the authored prompt. */
  range: { end: number; start: number };
}

const WILDCARD_GLOB_CHARACTER_RE = /[*?[]/;
const WILDCARD_REFERENCE_PATH_RE = /^[A-Za-z0-9_/*?[\]!-]+$/;
const isWildcardBoundaryCharacter = (character: string | undefined): boolean =>
  character !== undefined && /[A-Za-z0-9_]/.test(character);

const getWildcardLookupPath = (content: string): string | null => {
  const withoutSampler = content[0] === '~' || content[0] === '@' ? content.slice(1) : content;
  const parametersStart = withoutSampler.indexOf('(');
  let lookupPath = withoutSampler;

  if (parametersStart >= 0) {
    if (
      !withoutSampler.endsWith(')') ||
      withoutSampler.slice(parametersStart + 1, -1).includes('(') ||
      withoutSampler.slice(parametersStart + 1, -1).includes(')')
    ) {
      return null;
    }

    lookupPath = withoutSampler.slice(0, parametersStart);
  } else if (withoutSampler.includes(')')) {
    return null;
  }

  if (!lookupPath || !WILDCARD_REFERENCE_PATH_RE.test(lookupPath)) {
    return null;
  }

  return WILDCARD_GLOB_CHARACTER_RE.test(lookupPath) || WILDCARD_NAME_RE.test(lookupPath) ? lookupPath : null;
};

/** Scans every backend-compatible wildcard reference without regex state or backtracking. */
export const scanWildcardReferences = (prompt: string): WildcardReference[] => {
  const references: WildcardReference[] = [];
  let index = 0;

  while (index < prompt.length - 1) {
    if (prompt[index] !== '_' || prompt[index + 1] !== '_' || isWildcardBoundaryCharacter(prompt[index - 1])) {
      index++;
      continue;
    }

    const closing = prompt.indexOf('__', index + 2);

    if (closing < 0) {
      break;
    }

    const lookupPath = getWildcardLookupPath(prompt.slice(index + 2, closing));

    if (lookupPath) {
      references.push({ lookupPath, range: { end: closing + 2, start: index } });
      index = closing + 2;
    } else {
      index += 2;
    }
  }

  return references;
};

type GlobToken =
  | { kind: 'any' }
  | { kind: 'star' }
  | { kind: 'literal'; value: string }
  | {
      invalid: boolean;
      kind: 'class';
      literals: Set<string>;
      negated: boolean;
      ranges: { end: string; start: string }[];
    };

const parseGlobClass = (pattern: string, start: number): { end: number; token: GlobToken } | null => {
  let cursor = start + 1;
  const negated = pattern[cursor] === '!';

  if (negated) {
    cursor++;
  }
  if (pattern[cursor] === ']') {
    cursor++;
  }

  const closing = pattern.indexOf(']', cursor);

  if (closing < 0) {
    return null;
  }

  const contentStart = start + 1 + (negated ? 1 : 0);
  const content = pattern.slice(contentStart, closing);
  const literals = new Set<string>();
  const ranges: { end: string; start: string }[] = [];
  let invalid = false;

  for (let index = 0; index < content.length; index++) {
    const character = content[index]!;
    const rangeEnd = content[index + 2];

    if (content[index + 1] === '-' && rangeEnd !== undefined) {
      if (character > rangeEnd) {
        invalid = true;
      } else {
        ranges.push({ end: rangeEnd, start: character });
      }
      index += 2;
    } else {
      literals.add(character);
    }
  }

  return {
    end: closing + 1,
    token: { invalid, kind: 'class', literals, negated, ranges },
  };
};

const tokenizeGlob = (pattern: string): GlobToken[] => {
  const tokens: GlobToken[] = [];
  let index = 0;

  while (index < pattern.length) {
    const character = pattern[index]!;

    if (character === '*') {
      if (tokens.at(-1)?.kind !== 'star') {
        tokens.push({ kind: 'star' });
      }
      index++;
    } else if (character === '?') {
      tokens.push({ kind: 'any' });
      index++;
    } else if (character === '[') {
      const parsed = parseGlobClass(pattern, index);

      if (parsed) {
        tokens.push(parsed.token);
        index = parsed.end;
      } else {
        tokens.push({ kind: 'literal', value: character });
        index++;
      }
    } else {
      tokens.push({ kind: 'literal', value: character });
      index++;
    }
  }

  return tokens;
};

const globTokenMatches = (token: Exclude<GlobToken, { kind: 'star' }>, character: string): boolean => {
  if (token.kind === 'any') {
    return true;
  }
  if (token.kind === 'literal') {
    return token.value === character;
  }

  const inClass =
    !token.invalid &&
    (token.literals.has(character) || token.ranges.some((range) => character >= range.start && character <= range.end));

  return token.negated ? !inClass : inClass;
};

/** Bounded O(pattern × name) matcher mirroring Python `fnmatchcase`. */
const matchesGlob = (pattern: string, name: string): boolean => {
  let reachable = Array.from({ length: name.length + 1 }, (_, index) => index === 0);

  for (const token of tokenizeGlob(pattern)) {
    const next = Array.from({ length: name.length + 1 }, () => false);

    if (token.kind === 'star') {
      let canReach = false;

      for (let index = 0; index <= name.length; index++) {
        canReach ||= reachable[index] === true;
        next[index] = canReach;
      }
    } else {
      for (let index = 0; index < name.length; index++) {
        next[index + 1] = reachable[index] === true && globTokenMatches(token, name[index]!);
      }
    }

    reachable = next;
  }

  return reachable[name.length] === true;
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
  if (!WILDCARD_GLOB_CHARACTER_RE.test(path)) {
    return knownNames.has(path);
  }

  for (const name of knownNames) {
    if (matchesGlob(path, name)) {
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
  /\{[\s\S]*\}/.test(prompt) || prompt.includes('#') || scanWildcardReferences(prompt).length > 0;

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
