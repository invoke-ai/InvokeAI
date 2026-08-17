import type { TabName } from 'features/ui/store/uiTypes';
import { type AnyModelConfigWithExternal, isExternalApiModelConfig } from 'services/api/types';

export const isExternalModelUnsupportedForTab = (model: AnyModelConfigWithExternal, tab: TabName): boolean => {
  if (!isExternalApiModelConfig(model)) {
    return false;
  }

  if (tab === 'generate') {
    return !model.capabilities.modes.includes('txt2img');
  }

  return false;
};

/**
 * Some type=main configs are not selectable primary models: they fill a secondary slot in a
 * family's advanced section, and selecting them as the main model always fails at load time.
 * Every picker that offers main models must filter with this predicate.
 */
export const isSecondarySlotMainModelConfig = (c: AnyModelConfigWithExternal): boolean => {
  // Low-noise Wan GGUFs belong in the Transformer (Low Noise) slot of the Wan advanced section,
  // not as a primary main - filter them out so users can't accidentally wire them backwards.
  if (c.type === 'main' && c.base === 'wan' && c.format === 'gguf_quantized' && 'expert' in c && c.expert === 'low') {
    return true;
  }
  // MiniMax H3 single-file transformers belong in the Transformer (single file) slot of the
  // MiniMax H3 advanced section - they carry no text encoder or VAEs.
  if (c.type === 'main' && c.base === 'minimax-h3' && c.format === 'checkpoint') {
    return true;
  }
  return false;
};
