import type { ControlAdapterKind } from '@features/generation/graph';
import type { ModelConfig } from '@features/models';

export const getCompatibleControlModels = (
  models: readonly ModelConfig[],
  base: string | null,
  kind: ControlAdapterKind
): ModelConfig[] => {
  if (kind === 'z_image_control' && base !== 'z-image') {
    return [];
  }
  const modelType = kind === 'z_image_control' ? 'controlnet' : kind;
  return models.filter((model) => model.type === modelType && (!base || model.base === base));
};

/**
 * The control model a freshly created layer should start with: a union model
 * (the generalist), then tile, then the first compatible model. Null when no
 * compatible model is installed.
 */
export const resolveDefaultControlModel = (
  models: readonly ModelConfig[],
  base: string | null,
  kind: ControlAdapterKind
): string | null => {
  const compatible = getCompatibleControlModels(models, base, kind);
  const preferred =
    compatible.find((model) => model.name.toLowerCase().includes('union')) ??
    compatible.find((model) => model.name.toLowerCase().includes('tile')) ??
    compatible[0];
  return preferred?.key ?? null;
};

/** Default model for the adapter kind the layer factories pick for `base`. */
export const resolveDefaultControlModelForBase = (models: readonly ModelConfig[], base: string | null): string | null =>
  resolveDefaultControlModel(models, base, base === 'z-image' ? 'z_image_control' : 'controlnet');
