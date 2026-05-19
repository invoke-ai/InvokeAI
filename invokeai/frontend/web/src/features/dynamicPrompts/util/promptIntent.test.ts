import { describe, expect, it } from 'vitest';

import {
  getHasMixedCyclicAndNonCyclicDynamicPromptSyntax,
  getHasNonCyclicDynamicPromptSyntax,
  getIsCycleOnlyDynamicPrompt,
} from './promptIntent';

describe('promptIntent', () => {
  it('detects cycle-only dynamic prompts', () => {
    expect(getIsCycleOnlyDynamicPrompt('portrait __@lighting/studio__')).toBe(true);
    expect(getIsCycleOnlyDynamicPrompt('portrait (__@lighting/studio__)++')).toBe(true);
  });

  it('detects mixed cyclic and non-cyclic dynamic prompts', () => {
    const prompt = '__@lighting/studio__, __camera/lens__';

    expect(getIsCycleOnlyDynamicPrompt(prompt)).toBe(false);
    expect(getHasNonCyclicDynamicPromptSyntax(prompt)).toBe(true);
    expect(getHasMixedCyclicAndNonCyclicDynamicPromptSyntax(prompt)).toBe(true);
  });

  it('treats braced variants as non-cyclic dynamic syntax', () => {
    expect(getHasNonCyclicDynamicPromptSyntax('__@lighting/studio__ {warm|cool}')).toBe(true);
  });
});
