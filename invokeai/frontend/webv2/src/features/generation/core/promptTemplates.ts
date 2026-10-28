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

export interface EffectivePrompts {
  negativePrompt: string;
  positivePrompt: string;
}

interface AuthoredPrompts {
  negativePrompt: string;
  positivePrompt: string;
  promptTemplate?: PromptTemplateSnapshot | null;
}

// A function replacement keeps `$` sequences in authored prompts literal.
export const applyPromptTemplate = (templatePrompt: string, prompt: string): string =>
  templatePrompt.includes(PROMPT_TEMPLATE_PLACEHOLDER)
    ? templatePrompt.replace(PROMPT_TEMPLATE_PLACEHOLDER, () => prompt)
    : `${prompt} ${templatePrompt}`;

export const getEffectivePrompts = (settings: AuthoredPrompts): EffectivePrompts => {
  const { negativePrompt, positivePrompt, promptTemplate } = settings;

  if (!promptTemplate) {
    return { negativePrompt, positivePrompt };
  }

  return {
    negativePrompt: applyPromptTemplate(promptTemplate.negativePrompt, negativePrompt),
    // Text around a JSON prompt (a MOSS-Image layout, a structured prompt) would stop it parsing, so it stays whole.
    positivePrompt: isJsonPrompt(positivePrompt)
      ? positivePrompt
      : applyPromptTemplate(promptTemplate.positivePrompt, positivePrompt),
  };
};

export const flattenPromptTemplateExpansion = ({
  authoredNegativePrompt,
  selectedPositivePrompt,
  template,
}: {
  authoredNegativePrompt: string;
  selectedPositivePrompt: string;
  template: PromptTemplateSnapshot;
}): {
  negativePrompt: string;
  positivePrompt: string;
  promptTemplate: null;
  promptTemplateViewMode: false;
} => ({
  negativePrompt: applyPromptTemplate(template.negativePrompt, authoredNegativePrompt),
  positivePrompt: selectedPositivePrompt,
  promptTemplate: null,
  promptTemplateViewMode: false,
});

export const getPromptTemplateChunks = (prompt: string, templatePrompt: string): [string, string, string] => {
  if (!templatePrompt.includes(PROMPT_TEMPLATE_PLACEHOLDER)) {
    return ['', `${prompt} `, templatePrompt];
  }

  const [before, ...after] = templatePrompt.split(PROMPT_TEMPLATE_PLACEHOLDER);

  return [before ?? '', prompt, after.join(PROMPT_TEMPLATE_PLACEHOLDER)];
};

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
