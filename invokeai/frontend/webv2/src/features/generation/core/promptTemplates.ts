export const PROMPT_TEMPLATE_PLACEHOLDER = '{prompt}';

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
    positivePrompt: applyPromptTemplate(promptTemplate.positivePrompt, positivePrompt),
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
