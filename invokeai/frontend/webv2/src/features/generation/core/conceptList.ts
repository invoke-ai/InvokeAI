import type { GenerateLora, GenerateModelConfig, LoraModelConfig, WanLoraTarget } from './types';

import { getDefaultLoraWeight, isLoraCompatibleWithModel } from './settings';

/**
 * Concept (LoRA) list edits shared by every widget that stores a `GenerateLora[]`. Adds prevent new duplicate keys and
 * only add concepts the main model can load. An edit that changes nothing returns the same array, so callers can skip
 * the write: a row removed mid-edit still flushes its draft, and that must stay a no-op.
 */

export interface ConceptUpdate {
  isEnabled?: boolean;
  /** `null` collapses a split low-noise weight back onto the uniform `weight`. */
  lowNoiseWeight?: number | null;
  /** `null` clears any recalled Wan expert routing (along with `lowNoiseWeight`). */
  target?: WanLoraTarget | null;
  weight?: number;
}

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
  let didUpdate = false;
  const next = loras.map((lora) => {
    if (lora.model.key !== key) {
      return lora;
    }

    const nextIsEnabled = update.isEnabled ?? lora.isEnabled;
    const nextWeight = update.weight ?? lora.weight;
    const nextTarget = update.target === undefined ? lora.target : (update.target ?? undefined);
    // Low-noise split weight is only valid when targeted across both Wan experts.
    const nextLowNoiseWeight =
      nextTarget !== 'both'
        ? undefined
        : update.lowNoiseWeight === undefined
          ? lora.lowNoiseWeight
          : (update.lowNoiseWeight ?? undefined);

    if (
      nextIsEnabled === lora.isEnabled &&
      nextWeight === lora.weight &&
      nextTarget === lora.target &&
      nextLowNoiseWeight === lora.lowNoiseWeight
    ) {
      return lora;
    }

    didUpdate = true;
    const { lowNoiseWeight: _clearedLow, target: _clearedTarget, ...base } = lora;
    return {
      ...base,
      isEnabled: nextIsEnabled,
      weight: nextWeight,
      ...(nextTarget !== undefined ? { target: nextTarget } : {}),
      ...(nextLowNoiseWeight !== undefined ? { lowNoiseWeight: nextLowNoiseWeight } : {}),
    };
  });

  return didUpdate ? next : loras;
};

export const removeConcept = (loras: GenerateLora[], key: string): GenerateLora[] => {
  const next = loras.filter((lora) => lora.model.key !== key);

  return next.length === loras.length ? loras : next;
};
