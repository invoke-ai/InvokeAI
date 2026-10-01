import { describe, expect, it } from 'vitest';

import { calculatePromptTokens, getTokenizerConfig } from './tokenizers';

describe('tokenizers', () => {
  describe('getTokenizerConfig', () => {
    it('returns CLIP tokenizer config for SD-1, SD-2, SDXL, FLUX', () => {
      expect(getTokenizerConfig('sd-1')).toEqual({ family: 'clip', limit: 77 });
      expect(getTokenizerConfig('sdxl')).toEqual({ family: 'clip', limit: 77 });
      expect(getTokenizerConfig('flux')).toEqual({ family: 'clip', limit: 77 });
    });

    it('returns Qwen config for FLUX2, Z-Image, Anima, Krea-2', () => {
      expect(getTokenizerConfig('z-image')).toEqual({ family: 'qwen', limit: 512 });
      expect(getTokenizerConfig('anima')).toEqual({ family: 'qwen', limit: 512 });
    });

    it('returns estimate config for unknown models', () => {
      expect(getTokenizerConfig(undefined)).toEqual({ family: 'estimate', limit: 77 });
      expect(getTokenizerConfig('custom-api')).toEqual({ family: 'estimate', limit: 77 });
    });
  });

  describe('calculatePromptTokens', () => {
    it('returns 0 count for empty prompt', () => {
      const res = calculatePromptTokens('', 'sd-1');
      expect(res.count).toBe(0);
      expect(res.isNearLimit).toBe(false);
      expect(res.isOverLimit).toBe(false);
    });

    it('counts CLIP tokens correctly including BOS/EOS', () => {
      const res = calculatePromptTokens('a cute cat sitting on a bench', 'sd-1');
      expect(res.count).toBeGreaterThan(2);
      expect(res.limit).toBe(77);
    });

    it('flags near limit and over limit correctly', () => {
      const longPrompt = Array(85).fill('word').join(' ');
      const res = calculatePromptTokens(longPrompt, 'sd-1');
      expect(res.isOverLimit).toBe(true);
    });
  });
});
