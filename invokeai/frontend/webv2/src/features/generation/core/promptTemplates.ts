/**
 * Prompt templates: named prompt fragments that wrap the authored prompt at
 * submit time.
 *
 * The authored prompt and the generated prompt are deliberately different
 * strings. The user edits and recalls what they typed; the model sees the
 * template's text with `{prompt}` replaced. `getEffectivePrompts` is the single
 * place that difference is computed, and every submit and preview path reads it
 * rather than merging inline.
 *
 * Order matters against dynamic prompts. `hasDynamicPromptSyntax` matches any
 * `{…}`, so an unmerged `{prompt}` would be handed to the expander and come back
 * as the literal word "prompt". Templates merge first, which consumes the
 * placeholder before the expander ever sees it — and lets a template carry its
 * own `{a|b}` or `__wildcard__` syntax.
 *
 * The active template is stored as a denormalized snapshot rather than an id, so
 * the pure submit reducer can resolve prompts with no catalog lookup and a queue
 * item still explains itself after its template is deleted.
 */

export const PROMPT_TEMPLATE_PLACEHOLDER = '{prompt}';

/** The active template, copied into GenerateSettings. */
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
  /**
   * Optional so a surface with no template concept — Upscale, whose settings
   * carry no such field — can be passed straight through and merge to identity.
   */
  promptTemplate?: PromptTemplateSnapshot | null;
}

/**
 * Ported verbatim from the legacy `buildPresetModifiedPrompt` so a template
 * written in either client reads the same: the placeholder is substituted (first
 * occurrence only), and a template without one is appended after a space.
 *
 * The replacement is a function rather than the prompt itself. Passing a string
 * makes `$$`, `$&`, `` $` `` and `$'` inside it special even against a string
 * pattern, so `a poster reading $$$` lost a character and `x $& y` re-inserted
 * the literal `{prompt}` — which then reached the expander and came back as the
 * word "prompt", the exact failure merging first is supposed to prevent. It also
 * has to stay literal for `getPromptTemplateChunks` below, which splits, to keep
 * describing what actually generates.
 */
export const applyPromptTemplate = (templatePrompt: string, prompt: string): string =>
  templatePrompt.includes(PROMPT_TEMPLATE_PLACEHOLDER)
    ? templatePrompt.replace(PROMPT_TEMPLATE_PLACEHOLDER, () => prompt)
    : `${prompt} ${templatePrompt}`;

/** The prompts that will actually generate. Identity when no template is active. */
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

/**
 * The merged prompt split around the authored text, for the read-only view mode:
 * the outer chunks are the template's own words and are dimmed, the middle chunk
 * is what the user typed.
 *
 * A template with several placeholders only substitutes at the first, so the
 * rest are rejoined into the trailing chunk.
 *
 * The chunks must always concatenate to `applyPromptTemplate`'s output, or the
 * preview would describe something other than what generates. Legacy's
 * `getViewModeChunks` special-cases an empty template back to the bare prompt
 * and so disagrees by a trailing space; that divergence is not reproduced here.
 */
export const getPromptTemplateChunks = (prompt: string, templatePrompt: string): [string, string, string] => {
  if (!templatePrompt.includes(PROMPT_TEMPLATE_PLACEHOLDER)) {
    return ['', `${prompt} `, templatePrompt];
  }

  const [before, ...after] = templatePrompt.split(PROMPT_TEMPLATE_PLACEHOLDER);

  return [before ?? '', prompt, after.join(PROMPT_TEMPLATE_PLACEHOLDER)];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * True when a stored value is already exactly a snapshot, so normalization can
 * hand back the same object instead of allocating an equal one. `null` is not
 * canonical here — the caller distinguishes it before asking.
 */
export const isCanonicalPromptTemplateSnapshot = (value: unknown): value is PromptTemplateSnapshot =>
  isRecord(value) &&
  typeof value.id === 'string' &&
  value.id.length > 0 &&
  typeof value.name === 'string' &&
  typeof value.negativePrompt === 'string' &&
  typeof value.positivePrompt === 'string' &&
  Object.keys(value).length === 4;

/** Reads an untrusted persisted/transported snapshot, or `null` when unusable. */
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

/**
 * Re-reads the stored snapshot from the live catalog, the way model snapshots are
 * re-read in `syncGenerateWidgetValuesWithModels`, so an edit made in another tab
 * or by another client lands without a bespoke subscription.
 *
 * Returns the *existing* object whenever nothing changed. The caller diffs with
 * `Object.is`, so allocating an equal snapshot each pass would commit forever.
 * A template that has left the catalog is kept rather than cleared: it is still
 * the truthful record of what this project generates with, and dropping it would
 * silently change the prompt while the catalog is merely still loading.
 */
export const syncPromptTemplateWithCatalog = (
  promptTemplate: PromptTemplateSnapshot | null,
  catalog: readonly PromptTemplateSnapshot[]
): PromptTemplateSnapshot | null => {
  if (!promptTemplate) {
    return null;
  }

  const current = catalog.find((template) => template.id === promptTemplate.id);

  return current && !areSnapshotsEqual(current, promptTemplate) ? current : promptTemplate;
};
