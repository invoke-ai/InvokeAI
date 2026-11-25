export const PROMPT_TEMPLATE_PLACEHOLDER = '{prompt}';

// Same pattern as the backend layout node: the whole prompt is one fenced block, as language models write JSON.
const PROMPT_FENCE = /^```[A-Za-z0-9_+-]*[ \t]*\r?\n([\s\S]*?)\r?\n?```$/;

/** The trimmed prompt, or the body of the one ``` block it consists of. */
export const unwrapPromptFence = (prompt: string): string => {
  const trimmed = prompt.trim();
  return PROMPT_FENCE.exec(trimmed)?.[1] ?? trimmed;
};

/**
 * Whether the prompt (unwrapped) is a JSON object or array, such as a MOSS-Image layout or a structured prompt. Its
 * braces are data, never dynamic-prompt syntax.
 */
export const isJsonPrompt = (prompt: string): boolean => {
  const body = unwrapPromptFence(prompt).trim();
  if (!body.startsWith('{') && !body.startsWith('[')) {
    return false;
  }
  try {
    const parsed: unknown = JSON.parse(body);
    return typeof parsed === 'object' && parsed !== null;
  } catch {
    return false;
  }
};

export interface PromptTemplateSnapshot {
  id: string;
  name: string;
  negativePrompt: string;
  positivePrompt: string;
}

/** What the user typed for a template's custom placeholders, keyed by lowercased name. Empty values are not stored. */
export type PromptTemplateFieldValues = Readonly<Record<string, string>>;

/** One identity for "no values", so an absent stored key and a normalized one compare equal by reference. */
export const EMPTY_PROMPT_TEMPLATE_FIELD_VALUES: PromptTemplateFieldValues = Object.freeze({});

export interface PromptTemplateField {
  /** The key its value is stored under. */
  name: string;
  label: string;
}

export interface EffectivePrompts {
  negativePrompt: string;
  positivePrompt: string;
}

/**
 * A template travels with the values of its fields: naming a template without them does not compile, so no caller
 * can render its fields empty by omission.
 */
export type PromptTemplateApplication =
  | { promptTemplate?: null; promptTemplateFieldValues?: PromptTemplateFieldValues }
  | { promptTemplate: PromptTemplateSnapshot | null; promptTemplateFieldValues: PromptTemplateFieldValues };

type AuthoredPrompts = { negativePrompt: string; positivePrompt: string } & PromptTemplateApplication;

type TemplatePart =
  | { kind: 'text'; text: string }
  | { kind: 'prompt' }
  | { kind: 'field'; name: string; spelling: string };

// The optional `$` marks a dynamic-prompt variable (`${name}`), which is not a field.
const TEMPLATE_TOKEN_RE = /\$?\{([A-Za-z][A-Za-z0-9_]*)\}/g;
const FIELD_VALUE_KEY_RE = /^[a-z][a-z0-9_]*$/;
const RESERVED_FIELD_NAME = 'prompt';

/**
 * Template text as literal text, the first `{prompt}` and `{name}` fields. Later `{prompt}` copies and other spellings
 * of it (`{Prompt}`) stay text, as they were before templates had fields.
 */
const parsePromptTemplate = (templatePrompt: string): TemplatePart[] => {
  const parts: TemplatePart[] = [];
  let textStart = 0;
  let hasPrompt = false;

  const pushText = (end: number) => {
    if (end > textStart) {
      parts.push({ kind: 'text', text: templatePrompt.slice(textStart, end) });
    }
  };

  for (const match of templatePrompt.matchAll(TEMPLATE_TOKEN_RE)) {
    const spelling = match[1]!;
    const isVariable = match[0].startsWith('$');
    const tokenStart = match.index + (isVariable ? 1 : 0);

    if (spelling === RESERVED_FIELD_NAME) {
      if (hasPrompt) {
        continue;
      }

      hasPrompt = true;
      pushText(tokenStart);
      parts.push({ kind: 'prompt' });
    } else if (isVariable || spelling.toLowerCase() === RESERVED_FIELD_NAME) {
      continue;
    } else {
      pushText(tokenStart);
      parts.push({ kind: 'field', name: spelling.toLowerCase(), spelling });
    }

    textStart = match.index + match[0].length;
  }

  pushText(templatePrompt.length);
  return parts;
};

const getFieldLabel = (spelling: string): string => {
  const words = spelling.replaceAll('_', ' ').trim();

  return words.charAt(0).toUpperCase() + words.slice(1);
};

/** The template's fields, positive side first, each once in order of first appearance. */
export const getPromptTemplateFields = (
  template: Pick<PromptTemplateSnapshot, 'negativePrompt' | 'positivePrompt'>
): PromptTemplateField[] => {
  const fields = new Map<string, PromptTemplateField>();

  for (const templatePrompt of [template.positivePrompt, template.negativePrompt]) {
    for (const part of parsePromptTemplate(templatePrompt)) {
      if (part.kind === 'field' && !fields.has(part.name)) {
        fields.set(part.name, { label: getFieldLabel(part.spelling), name: part.name });
      }
    }
  }

  return [...fields.values()];
};

const isStoredFieldValue = (name: string, value: unknown): value is string =>
  typeof value === 'string' && value !== '' && name !== RESERVED_FIELD_NAME && FIELD_VALUE_KEY_RE.test(name);

/**
 * Reads untrusted stored values. Canonical values keep their identity; none at all is the shared empty constant. Keys
 * are sorted, so equal values serialize alike whatever order they were typed in.
 */
export const normalizePromptTemplateFieldValues = (value: unknown): PromptTemplateFieldValues => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return EMPTY_PROMPT_TEMPLATE_FIELD_VALUES;
  }

  const entries = Object.entries(value);
  const valid = entries.filter(([name, entry]) => isStoredFieldValue(name, entry));

  if (valid.length === 0) {
    return EMPTY_PROMPT_TEMPLATE_FIELD_VALUES;
  }

  const isCanonical =
    valid.length === entries.length && entries.every(([name], index) => index === 0 || entries[index - 1]![0] < name);

  return isCanonical
    ? (value as PromptTemplateFieldValues)
    : Object.fromEntries(valid.sort(([left], [right]) => (left < right ? -1 : 1)));
};

/** A field's own value: a name such as `constructor` must not read what every object inherits. */
export const getPromptTemplateFieldValue = (values: PromptTemplateFieldValues, name: string): string =>
  Object.hasOwn(values, name) ? values[name]! : '';

export const setPromptTemplateFieldValue = (
  values: PromptTemplateFieldValues,
  name: string,
  value: string
): PromptTemplateFieldValues => normalizePromptTemplateFieldValues({ ...values, [name]: value });

/** The values a newly chosen template can use: those of the fields it shares with what was typed before. */
export const retainPromptTemplateFieldValues = (
  values: PromptTemplateFieldValues,
  template: Pick<PromptTemplateSnapshot, 'negativePrompt' | 'positivePrompt'>
): PromptTemplateFieldValues => {
  const names = new Set(getPromptTemplateFields(template).map((field) => field.name));
  const entries = Object.entries(values);
  const kept = entries.filter(([name]) => names.has(name));

  if (kept.length === entries.length) {
    return values;
  }

  return kept.length === 0 ? EMPTY_PROMPT_TEMPLATE_FIELD_VALUES : Object.fromEntries(kept);
};

/** A piece of the merged prompt: the template's own text, the authored prompt, or a field's value. */
export interface PromptTemplateSegment {
  kind: 'field' | 'prompt' | 'template';
  text: string;
}

/**
 * The merged prompt in order. Every value is placed in one pass over the template, so a prompt or field value that
 * itself contains `{prompt}` or `{name}` stays as typed. An unfilled field leaves nothing behind.
 */
export const getPromptTemplateSegments = (
  prompt: string,
  templatePrompt: string,
  fieldValues: PromptTemplateFieldValues
): PromptTemplateSegment[] => {
  const parts = parsePromptTemplate(templatePrompt);
  const segments: PromptTemplateSegment[] = [];

  // A template without the placeholder follows the prompt.
  if (!parts.some((part) => part.kind === 'prompt')) {
    segments.push({ kind: 'prompt', text: `${prompt} ` });
  }

  for (const part of parts) {
    const segment: PromptTemplateSegment =
      part.kind === 'text'
        ? { kind: 'template', text: part.text }
        : part.kind === 'prompt'
          ? { kind: 'prompt', text: prompt }
          : { kind: 'field', text: getPromptTemplateFieldValue(fieldValues, part.name) };

    if (segment.text) {
      segments.push(segment);
    }
  }

  return segments;
};

export const joinPromptTemplateSegments = (segments: readonly PromptTemplateSegment[]): string =>
  segments.map((segment) => segment.text).join('');

export const applyPromptTemplate = (
  templatePrompt: string,
  prompt: string,
  fieldValues: PromptTemplateFieldValues
): string => joinPromptTemplateSegments(getPromptTemplateSegments(prompt, templatePrompt, fieldValues));

/**
 * The template's positive side as it applies to `prompt`. Text around a JSON prompt (a MOSS-Image layout, a structured
 * prompt) would stop it parsing, so a JSON prompt gets the bare placeholder and stays whole.
 */
const getPositiveTemplatePrompt = (template: PromptTemplateSnapshot, prompt: string): string =>
  isJsonPrompt(prompt) ? PROMPT_TEMPLATE_PLACEHOLDER : template.positivePrompt;

/**
 * The positive prompt that generates: the authored prompt inside the template's positive side. Submission, the prompt
 * box's merged view and dynamic-prompt previews and counts all read it, so they cannot disagree. Layout-draft
 * validation checks it and the authored prompt, as template text around a broken draft would hide it.
 */
export const getEffectivePositivePrompt = (
  prompt: string,
  template: PromptTemplateSnapshot | null,
  fieldValues: PromptTemplateFieldValues
): string =>
  template ? applyPromptTemplate(getPositiveTemplatePrompt(template, prompt), prompt, fieldValues) : prompt;

export const getEffectivePrompts = (settings: AuthoredPrompts): EffectivePrompts => {
  const { negativePrompt, positivePrompt, promptTemplate } = settings;

  if (!promptTemplate) {
    return { negativePrompt, positivePrompt };
  }

  const fieldValues = settings.promptTemplateFieldValues ?? EMPTY_PROMPT_TEMPLATE_FIELD_VALUES;

  return {
    negativePrompt: applyPromptTemplate(promptTemplate.negativePrompt, negativePrompt, fieldValues),
    positivePrompt: getEffectivePositivePrompt(positivePrompt, promptTemplate, fieldValues),
  };
};

export const flattenPromptTemplateExpansion = ({
  authoredNegativePrompt,
  fieldValues,
  selectedPositivePrompt,
  template,
}: {
  authoredNegativePrompt: string;
  fieldValues: PromptTemplateFieldValues;
  selectedPositivePrompt: string;
  template: PromptTemplateSnapshot;
}): {
  negativePrompt: string;
  positivePrompt: string;
  promptTemplate: null;
  promptTemplateViewMode: false;
} => ({
  negativePrompt: applyPromptTemplate(template.negativePrompt, authoredNegativePrompt, fieldValues),
  positivePrompt: selectedPositivePrompt,
  promptTemplate: null,
  promptTemplateViewMode: false,
});

/** {@link getEffectivePositivePrompt} in its pieces, for the merged view. */
export const getPositivePromptTemplateSegments = (
  prompt: string,
  template: PromptTemplateSnapshot,
  fieldValues: PromptTemplateFieldValues
): PromptTemplateSegment[] =>
  getPromptTemplateSegments(prompt, getPositiveTemplatePrompt(template, prompt), fieldValues);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const SNAPSHOT_FIELDS = {
  id: true,
  name: true,
  negativePrompt: true,
  positivePrompt: true,
} satisfies Record<keyof PromptTemplateSnapshot, true>;

const SNAPSHOT_FIELD_NAMES = Object.keys(SNAPSHOT_FIELDS);

export const isCanonicalPromptTemplateSnapshot = (value: unknown): value is PromptTemplateSnapshot =>
  isRecord(value) &&
  Object.keys(value).length === SNAPSHOT_FIELD_NAMES.length &&
  SNAPSHOT_FIELD_NAMES.every((field) => typeof value[field] === 'string') &&
  value.id !== '';

export const sanitizePromptTemplateSnapshot = (value: unknown): PromptTemplateSnapshot | null => {
  if (!isRecord(value)) {
    return null;
  }

  const { id, name, negativePrompt, positivePrompt } = value;

  if (typeof id !== 'string' || !id || typeof name !== 'string') {
    return null;
  }

  return {
    id,
    name,
    negativePrompt: typeof negativePrompt === 'string' ? negativePrompt : '',
    positivePrompt: typeof positivePrompt === 'string' ? positivePrompt : '',
  };
};

const areSnapshotsEqual = (left: PromptTemplateSnapshot, right: PromptTemplateSnapshot): boolean =>
  left.id === right.id &&
  left.name === right.name &&
  left.negativePrompt === right.negativePrompt &&
  left.positivePrompt === right.positivePrompt;

// Keep a deleted catalog entry: its snapshot still describes what the project generates.
export const syncPromptTemplateWithCatalog = (
  promptTemplate: PromptTemplateSnapshot | null,
  catalog: readonly PromptTemplateSnapshot[]
): PromptTemplateSnapshot | null => {
  if (!promptTemplate) {
    return null;
  }

  const current = catalog.find((template) => template.id === promptTemplate.id);

  if (!current || areSnapshotsEqual(current, promptTemplate)) {
    return promptTemplate;
  }

  return {
    id: current.id,
    name: current.name,
    negativePrompt: current.negativePrompt,
    positivePrompt: current.positivePrompt,
  };
};
