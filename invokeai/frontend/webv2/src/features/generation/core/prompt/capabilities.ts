import { COMPEL_PROMPT_NODE_TYPES } from './compel';

/** The prompt syntax consumed by the selected model's text encoder. */
export type PromptSyntaxProfile = 'none' | 'compel' | 'anima';

export const getPromptSyntaxProfile = (base: string | null | undefined): PromptSyntaxProfile => {
  if (base === 'anima') {
    return 'anima';
  }

  return typeof base === 'string' && Object.hasOwn(COMPEL_PROMPT_NODE_TYPES, base) ? 'compel' : 'none';
};

export const supportsPromptAttentionSyntax = (base: string | null | undefined): boolean =>
  getPromptSyntaxProfile(base) !== 'none';

/** Method tails belong to Compel and must stay plain and uneditable for Anima. */
export const getUnsupportedPromptMethodRanges = (prompt: string): { start: number; end: number }[] =>
  Array.from(prompt.matchAll(/\.[A-Za-z_][A-Za-z0-9_]*\([^()]*\)/g), (match) => ({
    start: match.index,
    end: match.index + match[0].length,
  }));
