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

const hasSystemDifference = (storedValues: unknown, values: GenerateWidgetValues): boolean => {
  if (!isRecord(storedValues)) {
    return true;
  }

  return Object.entries(values).some(
    ([key, value]) => key !== 'batchCount' && !areJsonValuesStructurallyEqual(storedValues[key], value)
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
  const selectedModel =
    supportedModels.find((model) => model.key === normalizedSettings?.modelKey) ?? supportedModels[0]!;
  const normalizedWidgetValues = normalizeGenerateWidgetValues(storedValues);
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
