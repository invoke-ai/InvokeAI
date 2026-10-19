import type { GenerateLora, GenerateModelConfig, LoraModelConfig } from './types';

import { getDefaultLoraWeight, isLoraCompatibleWithModel } from './settings';

/**
 * Concept (LoRA) list edits shared by every widget that stores a `GenerateLora[]`. Each list holds a model key at
 * most once and only gains concepts its main model can load. An edit that changes nothing returns the same array,
 * so callers can skip the write: a row removed mid-edit still flushes its draft, and that must stay a no-op.
 */

export type ConceptUpdate = Partial<Pick<GenerateLora, 'isEnabled' | 'weight'>>;

/** Appends an enabled concept at its default weight unless it is already listed or `mainModel` cannot load it. */
export const addConcept = (
  loras: GenerateLora[],
  model: LoraModelConfig,
  mainModel: Pick<GenerateModelConfig, 'base' | 'variant'> | null | undefined
): GenerateLora[] =>
  !mainModel || !isLoraCompatibleWithModel(model, mainModel) || loras.some((lora) => lora.model.key === model.key)
    ? loras
    : [...loras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }];

export const updateConcept = (loras: GenerateLora[], key: string, update: ConceptUpdate): GenerateLora[] => {
  const current = loras.find((lora) => lora.model.key === key);

  if (
    !current ||
    ((update.isEnabled === undefined || update.isEnabled === current.isEnabled) &&
      (update.weight === undefined || update.weight === current.weight))
  ) {
    return loras;
  }

  return loras.map((lora) => (lora === current ? { ...current, ...update } : lora));
};

export const removeConcept = (loras: GenerateLora[], key: string): GenerateLora[] => {
  const next = loras.filter((lora) => lora.model.key !== key);

  return next.length === loras.length ? loras : next;
};
