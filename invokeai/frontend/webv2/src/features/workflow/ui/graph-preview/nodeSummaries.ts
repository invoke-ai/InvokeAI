import type { TFunction } from 'i18next';

/** Return null when no type-specific summary exists so callers use the generic node/input summary. */
export const getNodeSubtitle = (
  node: { id: string; type: string; inputs: Record<string, unknown> },
  t: TFunction
): string | null => {
  if (node.type === 'denoise_latents') {
    const { cfg_scale: cfg, steps } = node.inputs;

    if (typeof cfg === 'number' && typeof steps === 'number') {
      return t('graphPreview.nodeSummary.denoise', { cfg, steps });
    }

    return null;
  }

  if (node.type === 'noise') {
    const { height, width } = node.inputs;

    if (typeof height === 'number' && typeof width === 'number') {
      return t('graphPreview.nodeSummary.noise', { height, width });
    }

    return null;
  }

  return null;
};
