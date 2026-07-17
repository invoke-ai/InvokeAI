import type { Project } from '@workbench/types';

import { getProjectWidgetValues } from '@workbench/widgetState';

export interface ProjectPromptDraft {
  negativePrompt: string;
  negativePromptEnabled: boolean;
  positivePrompt: string;
}

export type ProjectPromptDraftPatch = Partial<ProjectPromptDraft>;

export const getPromptDraftFromValues = (values: Record<string, unknown>): ProjectPromptDraft => ({
  negativePrompt: typeof values.negativePrompt === 'string' ? values.negativePrompt : '',
  negativePromptEnabled: values.negativePromptEnabled !== false,
  positivePrompt: typeof values.positivePrompt === 'string' ? values.positivePrompt : '',
});

export const getProjectPromptDraft = (project: Project): ProjectPromptDraft =>
  getPromptDraftFromValues(getProjectWidgetValues(project, 'generate'));

export const areProjectPromptDraftsEqual = (left: ProjectPromptDraft, right: ProjectPromptDraft): boolean =>
  left.positivePrompt === right.positivePrompt &&
  left.negativePrompt === right.negativePrompt &&
  left.negativePromptEnabled === right.negativePromptEnabled;

export const applyProjectPromptDraft = (
  values: Record<string, unknown>,
  patch: ProjectPromptDraftPatch
): Record<string, unknown> => {
  const current = getPromptDraftFromValues(values);
  const next: ProjectPromptDraft = {
    negativePrompt: typeof patch.negativePrompt === 'string' ? patch.negativePrompt : current.negativePrompt,
    negativePromptEnabled:
      typeof patch.negativePromptEnabled === 'boolean' ? patch.negativePromptEnabled : current.negativePromptEnabled,
    positivePrompt: typeof patch.positivePrompt === 'string' ? patch.positivePrompt : current.positivePrompt,
  };

  return areProjectPromptDraftsEqual(current, next) ? values : { ...values, ...next };
};

const hasPromptContent = (draft: ProjectPromptDraft): boolean =>
  draft.positivePrompt.trim().length > 0 || draft.negativePrompt.trim().length > 0;

/**
 * Generate owns the project prompt draft. Older projects may have prompt text stored only in
 * Upscale, so seed Generate from it when Generate has no prompt content of its own.
 */
export const migrateProjectPromptDraft = (
  generateValues: Record<string, unknown>,
  legacyUpscaleValues: Record<string, unknown>
): Record<string, unknown> => {
  const generateDraft = getPromptDraftFromValues(generateValues);
  const legacyUpscaleDraft = getPromptDraftFromValues(legacyUpscaleValues);

  if (hasPromptContent(generateDraft) || !hasPromptContent(legacyUpscaleDraft)) {
    return generateValues;
  }

  return applyProjectPromptDraft(generateValues, legacyUpscaleDraft);
};
