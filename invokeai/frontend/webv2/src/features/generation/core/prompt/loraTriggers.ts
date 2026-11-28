import type { PromptRange } from './ast';

export interface LoraTriggerPhrase {
  /** The LoRA's place in the concept list, so the prompt and the concept rows agree on its colour without storing one. */
  colorIndex: number;
  phrase: string;
}

/** Trimmed, non-empty and distinct, as both the prompt highlight and the concept rows show them. */
export const normalizeTriggerPhrases = (phrases: readonly string[] | null | undefined): string[] => [
  ...new Set((phrases ?? []).map((phrase) => phrase.trim()).filter(Boolean)),
];

/**
 * Each active LoRA's trigger phrases, longest first so a phrase wins over a shorter one starting at the same place.
 * `activePhrases` returns a LoRA's phrases, or nothing when the LoRA will not run.
 */
export const getLoraTriggerPhrases = <T>(
  loras: readonly T[],
  activePhrases: (lora: T) => readonly string[] | null | undefined
): LoraTriggerPhrase[] =>
  loras
    .flatMap((lora, index) =>
      normalizeTriggerPhrases(activePhrases(lora)).map((phrase) => ({ colorIndex: index, phrase }))
    )
    .sort((left, right) => right.phrase.length - left.phrase.length);

const escapeRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// The prompt tokenizer's word characters, so a match always covers whole tokens.
const WORD_CHAR = '[\\p{L}\\p{M}\\p{N}_]';

/** Case-insensitive whole-word matches: `cat` matches in `a cat,` but not in `catgirl`. */
export const findTriggerPhrase = (prompt: string, phrase: string): PromptRange[] => {
  const pattern = new RegExp(`(?<!${WORD_CHAR})${escapeRegExp(phrase)}(?!${WORD_CHAR})`, 'giu');

  return [...prompt.matchAll(pattern)].map((match) => ({ start: match.index, end: match.index + match[0].length }));
};
