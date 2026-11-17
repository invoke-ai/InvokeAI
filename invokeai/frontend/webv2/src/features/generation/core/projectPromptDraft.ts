import type { PromptTemplateSnapshot } from './promptTemplates';
import type { DynamicPromptsSettings } from './types';

import { isCanonicalPromptTemplateSnapshot, sanitizePromptTemplateSnapshot } from './promptTemplates';
import { normalizeDynamicPromptsSettings } from './settings';

/**
 * The prompt Generate and Upscale share, stored in Generate's values: the text, the template that wraps it, and how
 * it expands. They travel together because a shared prompt that generated differently on each surface would not be
 * shared.
 */
export interface ProjectPromptDraft extends DynamicPromptsSettings {
  negativePrompt: string;
  negativePromptEnabled: boolean;
  positivePrompt: string;
  promptTemplate: PromptTemplateSnapshot | null;
  /** Whether the prompt boxes show the template-merged prompt instead of the authored text. */
  promptTemplateViewMode: boolean;
}

export type ProjectPromptDraftPatch = Partial<ProjectPromptDraft>;

const TEXT_KEYS = ['negativePrompt', 'negativePromptEnabled', 'positivePrompt'] as const;
const DRAFT_KEYS = [
  ...TEXT_KEYS,
  'promptTemplate',
  'promptTemplateViewMode',
  'dynamicPromptsCombinatorial',
  'dynamicPromptsEnabled',
  'dynamicPromptsMaxPrompts',
  'dynamicPromptsResample',
  'dynamicPromptsSampleSeed',
  'dynamicPromptsSeedBehaviour',
] as const satisfies readonly (keyof ProjectPromptDraft)[];

/** Read as Generate normalizes the same fields, so both surfaces see one prompt. */
export const getPromptDraftFromValues = (values: Record<string, unknown>): ProjectPromptDraft => {
  // Keep a canonical snapshot's identity so repeated reads compare cheaply.
  const promptTemplate = isCanonicalPromptTemplateSnapshot(values.promptTemplate)
    ? values.promptTemplate
    : sanitizePromptTemplateSnapshot(values.promptTemplate);

  return {
    ...normalizeDynamicPromptsSettings(values),
    negativePrompt: typeof values.negativePrompt === 'string' ? values.negativePrompt : '',
    negativePromptEnabled: values.negativePromptEnabled !== false,
    positivePrompt: typeof values.positivePrompt === 'string' ? values.positivePrompt : '',
    promptTemplate,
    promptTemplateViewMode: promptTemplate !== null && values.promptTemplateViewMode === true,
  };
};

const arePromptTemplatesEqual = (left: PromptTemplateSnapshot | null, right: PromptTemplateSnapshot | null) =>
  left === right ||
  (left !== null &&
    right !== null &&
    left.id === right.id &&
    left.name === right.name &&
    left.negativePrompt === right.negativePrompt &&
    left.positivePrompt === right.positivePrompt);

export const areProjectPromptDraftsEqual = (left: ProjectPromptDraft, right: ProjectPromptDraft): boolean =>
  DRAFT_KEYS.every((key) =>
    key === 'promptTemplate'
      ? arePromptTemplatesEqual(left.promptTemplate, right.promptTemplate)
      : Object.is(left[key], right[key])
  );

/** Writes the text fields and whichever others the patch names, each read as the draft reads it. */
export const applyProjectPromptDraft = (
  values: Record<string, unknown>,
  patch: ProjectPromptDraftPatch
): Record<string, unknown> => {
  const current = getPromptDraftFromValues(values);
  const patchedKeys = DRAFT_KEYS.filter((key) => patch[key] !== undefined);
  const next = getPromptDraftFromValues({
    ...current,
    ...Object.fromEntries(patchedKeys.map((key) => [key, patch[key]])),
  });

  if (areProjectPromptDraftsEqual(current, next)) {
    return values;
  }

  const written = new Set<keyof ProjectPromptDraft>([...TEXT_KEYS, ...patchedKeys]);

  // Clearing the template also leaves its merged view.
  if (written.has('promptTemplate')) {
    written.add('promptTemplateViewMode');
  }

  return { ...values, ...Object.fromEntries([...written].map((key) => [key, next[key]])) };
};

const hasPromptContent = (draft: ProjectPromptDraft): boolean =>
  draft.positivePrompt.trim().length > 0 || draft.negativePrompt.trim().length > 0;

/** Seed Generate's prompt text from Upscale only when Generate has none; Upscale never stored the rest. */
export const migrateProjectPromptDraft = (
  generateValues: Record<string, unknown>,
  legacyUpscaleValues: Record<string, unknown>
): Record<string, unknown> => {
  const generateDraft = getPromptDraftFromValues(generateValues);
  const legacyUpscaleDraft = getPromptDraftFromValues(legacyUpscaleValues);

  if (hasPromptContent(generateDraft) || !hasPromptContent(legacyUpscaleDraft)) {
    return generateValues;
  }

  return applyProjectPromptDraft(
    generateValues,
    Object.fromEntries(TEXT_KEYS.map((key) => [key, legacyUpscaleDraft[key]]))
  );
};
