import type { ModelConfig } from '@features/models';

import { getControlModelUnusableReason } from '@features/generation/graph';

/** Anima LLLite models from before `cond_in_channels` was recorded: listed disabled, never usable. */
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
