import { hasArchitectureCapabilities } from '@features/generation/core/architectureCapabilities';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';

import type { GenerationModelCatalogItem } from './contracts';
import type { PromptTemplateSnapshot } from './promptTemplates';
import type { GenerateWidgetValues } from './types';

import {
  getAutoFlux2ComponentSourceModel,
  getDefaultGenerateSettings,
  isArchitectureDescribed,
  isGenerateModelSelectable,
} from './baseGenerationPolicies';
import { syncPromptTemplateWithCatalog } from './promptTemplates';
import {
  isGenerateWidgetValues,
  isSameModelFile,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  syncGenerateWidgetValuesWithModels,
} from './settings';

export interface ResolveGenerateWidgetValuesInput {
  models: readonly GenerationModelCatalogItem[];
  /**
   * Absent while the catalog is unread or failed. A successful empty catalog
   * is distinct, but both preserve an applied snapshot that is not present.
   */
  promptTemplates?: readonly PromptTemplateSnapshot[];
  storedValues: unknown;
}

export interface ResolvedGenerateWidgetValues {
  /** Canonical values for rendering and user commits. */
  values: GenerateWidgetValues;
  /** A fixed point returns null; batchCount belongs to the topbar and is excluded. */
  systemPatch: Partial<GenerateWidgetValues> | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object';

/**
 * Fields newer than some saved projects whose absence already reads as their default, so loading such a project does
 * not write them in: the project stays as saved until the user changes the setting.
 */
const DEFAULTS_READ_FROM_ABSENCE: Readonly<Partial<Record<keyof GenerateWidgetValues, unknown>>> = {
  dynamicPromptsEnabled: true,
};

const hasSystemDifference = (storedValues: unknown, values: GenerateWidgetValues): boolean => {
  if (!isRecord(storedValues)) {
    return true;
  }

  return Object.entries(values).some(
    ([key, value]) =>
      key !== 'batchCount' &&
      !(storedValues[key] === undefined && DEFAULTS_READ_FROM_ABSENCE[key as keyof GenerateWidgetValues] === value) &&
      !areJsonValuesStructurallyEqual(storedValues[key], value)
  );
};

const createSystemPatch = (values: GenerateWidgetValues): Partial<GenerateWidgetValues> => {
  const systemPatch: Partial<GenerateWidgetValues> = { ...values };

  delete systemPatch.batchCount;
  return systemPatch;
};

/** This resolver owns reconciliation; applying systemPatch must reach a fixed point. */
export const resolveGenerateWidgetValues = ({
  models,
  promptTemplates,
  storedValues,
}: ResolveGenerateWidgetValuesInput): ResolvedGenerateWidgetValues | null => {
  // Gate reconciliation to prevent persisting fallback defaults.
  if (!hasArchitectureCapabilities()) {
    return null;
  }

  // Require a described architecture before selecting defaults for persistence.
  const supportedModels = models.filter(isGenerateModelSelectable).filter(isArchitectureDescribed);

  if (supportedModels.length === 0) {
    return null;
  }

  const normalizedSettings = normalizeGenerateSettings(storedValues);
  const normalizedWidgetValues = normalizeGenerateWidgetValues(storedValues);
  const storedModel = normalizedWidgetValues?.model;
  // A chosen model that is no longer installed stays chosen: the model card names it and Invoke waits until the user
  // picks another or it returns. Reinstalling gives it a new key but the same file hash, so it is found again by hash
  // (and its type, base and variant).
  // Only values with no runnable choice (first run, an unknown family) take the default.
  const selectedModel =
    supportedModels.find((model) => model.key === normalizedSettings?.modelKey) ??
    (storedModel ? supportedModels.find((model) => isSameModelFile(model, storedModel)) : undefined) ??
    (storedModel &&
    storedModel.key === normalizedSettings?.modelKey &&
    isGenerateModelSelectable(storedModel) &&
    isArchitectureDescribed(storedModel)
      ? storedModel
      : supportedModels[0]!);
  const canReuseStoredValues =
    isGenerateWidgetValues(storedValues) && normalizedWidgetValues?.batchCount === storedValues.batchCount;
  const baseValues: GenerateWidgetValues = canReuseStoredValues
    ? storedValues
    : {
        ...(normalizedSettings ?? getDefaultGenerateSettings(selectedModel)),
        model: selectedModel,
      };

  let values = syncGenerateWidgetValuesWithModels(baseValues, models);

  if (values.model.key !== selectedModel.key || values.modelKey !== selectedModel.key) {
    values = { ...values, model: selectedModel, modelKey: selectedModel.key };
  }

  if (promptTemplates !== undefined) {
    const promptTemplate = syncPromptTemplateWithCatalog(values.promptTemplate, promptTemplates);

    if (promptTemplate !== values.promptTemplate) {
      values = { ...values, promptTemplate };
    }
  }

  const componentSourceModel = getAutoFlux2ComponentSourceModel(selectedModel, values, models);

  if (componentSourceModel !== undefined && componentSourceModel?.key !== values.componentSourceModel?.key) {
    values = { ...values, componentSourceModel };
  }

  return {
    systemPatch: hasSystemDifference(storedValues, values) ? createSystemPatch(values) : null,
    values,
  };
};
