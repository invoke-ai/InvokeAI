import type { SupportedGenerateBase } from '@features/generation/core/supportedBases';

/** The Compel node that encodes each SD base's prompts. */
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

/** Every Compel encoder node type, for graph checks that must not drift from this table. */
export const COMPEL_NODE_TYPES: ReadonlySet<string> = new Set([
  ...Object.values(COMPEL_PROMPT_NODE_TYPES),
  SDXL_REFINER_COMPEL_PROMPT_NODE_TYPE,
]);

const isCompelPromptBase = (base: string): base is CompelPromptBase => Object.hasOwn(COMPEL_PROMPT_NODE_TYPES, base);

/** For a graph path dispatched by base: throws rather than emit a conditioning node with no type. */
export const getCompelPromptNodeType = (base: string): (typeof COMPEL_PROMPT_NODE_TYPES)[CompelPromptBase] => {
  if (!isCompelPromptBase(base)) {
    throw new Error(`${base} prompts are not encoded with Compel.`);
  }

  return COMPEL_PROMPT_NODE_TYPES[base];
};

/** The prompt syntax consumed by the selected model's text encoder. */
export type PromptSyntaxProfile = 'none' | 'compel' | 'anima';

export const getPromptSyntaxProfile = (base: string | null | undefined): PromptSyntaxProfile => {
  if (base === 'anima') {
    return 'anima';
  }

  return typeof base === 'string' && isCompelPromptBase(base) ? 'compel' : 'none';
};

export const supportsPromptAttentionSyntax = (base: string | null | undefined): boolean =>
  getPromptSyntaxProfile(base) !== 'none';
