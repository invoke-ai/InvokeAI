import type { GenerateLora } from '@features/generation/core/types';

import { describe, expect, it } from 'vitest';

import { buildPromptHighlightSegments, type PromptHighlightOptions } from './highlight';
import { findTriggerPhrase, getLoraTriggerPhrases } from './loraTriggers';

const lora = (key: string, triggerPhrases: string[], isEnabled = true): GenerateLora => ({
  isEnabled,
  model: { base: 'sdxl', key, name: key, trigger_phrases: triggerPhrases, type: 'lora' } as GenerateLora['model'],
  weight: 1,
});

const triggerSegments = (prompt: string, loras: GenerateLora[], options: PromptHighlightOptions = {}) =>
  buildPromptHighlightSegments(prompt, {
    ...options,
    loraTriggers: getLoraTriggerPhrases(loras, (entry) => (entry.isEnabled ? entry.model.trigger_phrases : null)),
  })
    .filter((segment) => segment.kind === 'loraTrigger')
    .map(({ colorIndex, text }) => ({ colorIndex, text }));

describe('LoRA trigger phrases', () => {
  it("colours each phrase by its LoRA's place in the list, skipping inactive LoRAs", () => {
    const loras = [lora('a', ['pinup']), lora('b', ['neon'], false), lora('c', ['ohwx man'])];

    expect(triggerSegments('Pinup photo of ohwx man in neon light', loras)).toEqual([
      { colorIndex: 0, text: 'Pinup' },
      { colorIndex: 2, text: 'ohwx man' },
    ]);
  });

  it('matches whole words only', () => {
    expect(findTriggerPhrase('catgirl, cat_ears, a cat.', 'cat')).toEqual([{ start: 21, end: 24 }]);
    expect(findTriggerPhrase('price $5 (x)', '$5 (x)')).toEqual([{ start: 6, end: 12 }]);
    // A decomposed accent is part of the word, as the prompt tokenizer reads it.
    expect(findTriggerPhrase('cafe\u0301 au lait', 'cafe')).toEqual([]);
  });

  it('lets a longer phrase keep the words a shorter one inside it would claim', () => {
    const loras = [lora('a', ['girl']), lora('b', ['girl power'])];

    expect(triggerSegments('girl power, girl', loras)).toEqual([
      { colorIndex: 1, text: 'girl power' },
      { colorIndex: 0, text: 'girl' },
    ]);
  });

  it('leaves comments and embeddings to their own highlight', () => {
    const loras = [lora('a', ['ink'])];

    expect(triggerSegments('<ink> # ink\nink', loras, { dynamicPrompts: true })).toEqual([
      { colorIndex: 0, text: 'ink' },
    ]);
  });

  it('colours triggers for models that read no attention syntax', () => {
    expect(triggerSegments('(ink)1.2', [lora('a', ['ink'])], { attentionSyntax: false })).toEqual([
      { colorIndex: 0, text: 'ink' },
    ]);
  });
});
