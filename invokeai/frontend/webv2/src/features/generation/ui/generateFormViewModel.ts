import type { GenerateSettings, GenerateWidgetValues } from '@features/generation/core/types';

import { createStableSelector } from '@platform/state/selectors';

const PROMPT_FORM_KEYS = [
  'negativePrompt',
  'negativePromptEnabled',
  'negativePromptHeightPx',
  'positivePrompt',
  'positivePromptHeightPx',
] as const satisfies readonly (keyof GenerateSettings)[];
const EXTERNAL_GENERATE_FORM_KEYS = [
  ...PROMPT_FORM_KEYS,
  'batchCount',
] as const satisfies readonly (keyof GenerateSettings)[];
const EXTERNAL_GENERATE_FORM_KEY_SET = new Set<string>(EXTERNAL_GENERATE_FORM_KEYS);

export const areGenerateFormValuesEqual = (left: Record<string, unknown>, right: Record<string, unknown>): boolean => {
  if (left === right) {
    return true;
  }

  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);

  for (const key of keys) {
    if (EXTERNAL_GENERATE_FORM_KEY_SET.has(key)) {
      continue;
    }

    if (!Object.is(left[key], right[key])) {
      return false;
    }
  }

  return true;
};

export interface ProjectGenerateFormValuesSelection {
  projectId: string;
  values: Record<string, unknown>;
}

export const areProjectGenerateFormValuesEqual = (
  left: ProjectGenerateFormValuesSelection,
  right: ProjectGenerateFormValuesSelection
): boolean => left.projectId === right.projectId && areGenerateFormValuesEqual(left.values, right.values);

export const createGenerateFormValuesSelector = () =>
  createStableSelector((values: Record<string, unknown>) => values, areGenerateFormValuesEqual);

export const getSettingsWithLatestPromptFields = (
  settings: GenerateSettings,
  latestSettings: GenerateSettings
): GenerateSettings => {
  let nextSettings = settings;

  for (const key of PROMPT_FORM_KEYS) {
    if (!Object.is(nextSettings[key], latestSettings[key])) {
      nextSettings = { ...nextSettings, [key]: latestSettings[key] };
    }
  }

  return nextSettings;
};

export const getGenerateFormCommitPatch = (
  settings: GenerateSettings & Partial<Pick<GenerateWidgetValues, 'model'>>
): Partial<GenerateWidgetValues> => {
  const patch: Partial<GenerateWidgetValues> = { ...settings };

  delete patch.batchCount;

  return patch;
};
