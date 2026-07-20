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
