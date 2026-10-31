import { describe, expect, it } from 'vitest';

import { getCompelPromptNodeType, supportsPromptAttentionSyntax } from './compel';

describe('supportsPromptAttentionSyntax', () => {
  it.each(['sd-1', 'sd-2', 'sdxl'])('reads attention syntax for %s', (base) => {
    expect(supportsPromptAttentionSyntax(base)).toBe(true);
  });

  it.each(['flux', 'krea-2', 'flux2', 'sd-3', 'cogview4', 'z-image', 'qwen-image', 'anima', 'wan', 'external'])(
    'sends %s prompts as plain text',
    (base) => {
      expect(supportsPromptAttentionSyntax(base)).toBe(false);
    }
  );

  it('reads nothing without a model or for inherited object keys', () => {
    expect(supportsPromptAttentionSyntax(undefined)).toBe(false);
    expect(supportsPromptAttentionSyntax(null)).toBe(false);
    expect(supportsPromptAttentionSyntax('constructor')).toBe(false);
  });
});

describe('getCompelPromptNodeType', () => {
  it('names each Compel base its own encoder node', () => {
    expect(getCompelPromptNodeType('sd-1')).toBe('compel');
    expect(getCompelPromptNodeType('sd-2')).toBe('compel');
    expect(getCompelPromptNodeType('sdxl')).toBe('sdxl_compel_prompt');
  });

  it('refuses a base dispatched to a Compel path by mistake', () => {
    expect(() => getCompelPromptNodeType('flux')).toThrow('flux prompts are not encoded with Compel.');
  });
});
