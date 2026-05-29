/**
 * Maps a FLUX.2 Klein main-model variant to the Qwen3 encoder variant it uses.
 * Multiple Klein variants can share the same Qwen3 variant (e.g. `klein_9b` and
 * `klein_9b_base` both use `qwen3_8b`), so two different Klein variants can be
 * Qwen3-compatible sources for each other.
 */
export const KLEIN_TO_QWEN3_VARIANT_MAP: Record<string, string> = {
  klein_4b: 'qwen3_4b',
  klein_4b_base: 'qwen3_4b',
  klein_9b: 'qwen3_8b',
  klein_9b_base: 'qwen3_8b',
};

/**
 * Returns true if two Klein variants share the same Qwen3 encoder and can therefore
 * be used as a Qwen3 source for each other.
 */
export const isFlux2KleinQwen3Compatible = (variantA: unknown, variantB: unknown): boolean => {
  if (typeof variantA !== 'string' || typeof variantB !== 'string') {
    return false;
  }
  const qwen3A = KLEIN_TO_QWEN3_VARIANT_MAP[variantA];
  const qwen3B = KLEIN_TO_QWEN3_VARIANT_MAP[variantB];
  return qwen3A !== undefined && qwen3A === qwen3B;
};
