import type { SupportedGenerateBase } from '@features/generation/core/supportedBases';

/**
 * The Compel node that encodes each base's prompts. Compel is the only encoder that reads attention syntax: weights,
 * groups, `.and()`/`.blend()` and `<embedding>`. Every other base, external generators included, hands its text
 * encoder the prompt as written.
 */
export const COMPEL_PROMPT_NODE_TYPES = {
  'sd-1': 'compel',
  'sd-2': 'compel',
  sdxl: 'sdxl_compel_prompt',
} as const satisfies Partial<Record<SupportedGenerateBase, string>>;

export type CompelPromptBase = keyof typeof COMPEL_PROMPT_NODE_TYPES;

/**
 * The SDXL Refiner's encoder, Compel as well: the refiner has no prompts of its own and re-encodes the SDXL prompts,
 * so attention syntax carries through its stage. It keys no base: the refiner only ever runs behind an SDXL model.
 */
export const SDXL_REFINER_COMPEL_PROMPT_NODE_TYPE = 'sdxl_refiner_compel_prompt';

/** Every node type that reads attention syntax, for graph checks that must not drift from this table. */
export const COMPEL_NODE_TYPES: ReadonlySet<string> = new Set([
  ...Object.values(COMPEL_PROMPT_NODE_TYPES),
  SDXL_REFINER_COMPEL_PROMPT_NODE_TYPE,
]);

/** No model, no reader: an unselected model interprets nothing. */
export const supportsPromptAttentionSyntax = (base: string | null | undefined): base is CompelPromptBase =>
  typeof base === 'string' && Object.hasOwn(COMPEL_PROMPT_NODE_TYPES, base);

/** For a graph path dispatched by base: throws rather than emit a conditioning node with no type. */
export const getCompelPromptNodeType = (base: string): (typeof COMPEL_PROMPT_NODE_TYPES)[CompelPromptBase] => {
  if (!supportsPromptAttentionSyntax(base)) {
    throw new Error(`${base} prompts are not encoded with Compel.`);
  }

  return COMPEL_PROMPT_NODE_TYPES[base];
};
