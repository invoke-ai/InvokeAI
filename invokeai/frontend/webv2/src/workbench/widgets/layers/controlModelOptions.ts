import type { ModelConfig } from '@features/models';

import { getControlModelUnusableReason } from '@features/generation/graph';

/**
 * Anima ControlNet-LLLite models installed before `cond_in_channels` was recorded. The Model row lists them disabled,
 * so users learn they exist and need re-identifying; they never count as usable models.
 */
export const getReidentifyPendingControlModels = (
  models: readonly ModelConfig[],
  base: string | null
): ModelConfig[] =>
  base === 'anima'
    ? models.filter(
        (model) =>
          model.base === base && getControlModelUnusableReason(model, 'anima_lllite') === 'lllite_channels_unknown'
      )
    : [];
