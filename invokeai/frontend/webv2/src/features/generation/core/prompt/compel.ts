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
