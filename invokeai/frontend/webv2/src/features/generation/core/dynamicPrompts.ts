const SEED_MAX = 4_294_967_295;

export const DYNAMIC_PROMPTS_MIN_PROMPTS = 1;
export const DYNAMIC_PROMPTS_MAX_PROMPTS = 10_000;
export const DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS = 100;

export type DynamicPromptsSeedBehaviour = 'per-iteration' | 'per-image';

export interface DynamicPromptsConfig {
  combinatorial: boolean;
  maxPrompts: number;
  sampleSeed: number;
  seedBehaviour: DynamicPromptsSeedBehaviour;
}

const WILDCARD_NAME_SEGMENT = '[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?';
const WILDCARD_NAME_SOURCE = `${WILDCARD_NAME_SEGMENT}(?:/${WILDCARD_NAME_SEGMENT})*`;

export const WILDCARD_NAME_RE = new RegExp(`^${WILDCARD_NAME_SOURCE}$`);

export interface WildcardReference {
  lookupPath: string;
  range: { end: number; start: number };
}

const WILDCARD_GLOB_CHARACTER_RE = /[*?[]/;
const WILDCARD_REFERENCE_PATH_RE = /^[A-Za-z0-9_/*?[\]!-]+$/;

export const MAX_WILDCARD_NAME_LENGTH = 128;

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

  if (!lookupPath || lookupPath.length > MAX_WILDCARD_NAME_LENGTH || !WILDCARD_REFERENCE_PATH_RE.test(lookupPath)) {
    return null;
  }

  return WILDCARD_GLOB_CHARACTER_RE.test(lookupPath) || WILDCARD_NAME_RE.test(lookupPath) ? lookupPath : null;
};

export const scanWildcardReferences = (prompt: string): WildcardReference[] => {
  const references: WildcardReference[] = [];
  let index = 0;

  while (index < prompt.length - 1) {
    if (prompt[index] !== '_' || prompt[index + 1] !== '_') {
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

export const MAX_WILDCARD_VALUES = 10_000;
export const MAX_WILDCARD_VALUE_LENGTH = 2_000;

export const getWildcardValuesError = (values: readonly string[]): 'tooManyValues' | 'valueTooLong' | null => {
  if (values.length > MAX_WILDCARD_VALUES) {
    return 'tooManyValues';
  }
  if (values.some((value) => value.length > MAX_WILDCARD_VALUE_LENGTH)) {
    return 'valueTooLong';
  }

  return null;
};

export const normalizeWildcardValues = (values: readonly string[]): string[] =>
  values.map((value) => value.split('#')[0]?.trim() ?? '').filter((value) => value.length > 0);

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
