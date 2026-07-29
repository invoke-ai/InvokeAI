import type { GenerationModelCatalogItem } from './contracts';
import type { PromptTemplateSnapshot } from './promptTemplates';
import type { GenerateWidgetValues } from './types';

import {
  getAutoFlux2ComponentSourceModel,
  getDefaultGenerateSettings,
  isSupportedGenerateModel,
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
  /**
   * The complete canonical system-owned patch, or null at the fixed point.
   * `batchCount` is deliberately omitted because the Workbench topbar owns it.
   */
  systemPatch: Partial<GenerateWidgetValues> | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object';

const areValuesEqual = (left: unknown, right: unknown): boolean => {
  if (Object.is(left, right)) {
    return true;
  }

  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => areValuesEqual(value, right[index]))
    );
  }

  if (!isRecord(left) || !isRecord(right)) {
    return false;
  }

  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);

  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key) => Object.hasOwn(right, key) && areValuesEqual(left[key], right[key]))
  );
};

const hasSystemDifference = (storedValues: unknown, values: GenerateWidgetValues): boolean => {
  if (!isRecord(storedValues)) {
    return true;
  }

  return Object.entries(values).some(
    ([key, value]) => key !== 'batchCount' && !areValuesEqual(storedValues[key], value)
  );
};

const createSystemPatch = (values: GenerateWidgetValues): Partial<GenerateWidgetValues> => {
  const systemPatch: Partial<GenerateWidgetValues> = { ...values };

  delete systemPatch.batchCount;
  return systemPatch;
};

/**
 * Resolves persisted Generate widget state against backend-owned catalogs.
 *
 * This is the sole owner of default-model selection and denormalized snapshot
 * reconciliation. Applying `systemPatch` reaches a fixed point: the next call
 * returns the same canonical values with a null patch.
 */
export const resolveGenerateWidgetValues = ({
  models,
  promptTemplates,
  storedValues,
}: ResolveGenerateWidgetValuesInput): ResolvedGenerateWidgetValues | null => {
  const supportedModels = models.filter(isSupportedGenerateModel);

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
