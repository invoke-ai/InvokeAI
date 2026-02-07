import { describe, expect, it } from 'vitest';

import { adjustPromptAttention } from './promptAttention';

describe('adjustPromptAttention', () => {
  describe('cross-boundary selection', () => {
    it('should split group and apply attention when selection spans from inside group to outside (increment)', () => {
      const prompt = '(a b)+ c';
      const result = adjustPromptAttention(prompt, 3, 8, 'increment');

      expect(result.prompt).toBe('(a b+ c)+');
    });

    it('should split group and apply attention when selection spans from inside group to outside (decrement)', () => {
      const prompt = '(a b)+ c';
      const result = adjustPromptAttention(prompt, 3, 8, 'decrement');

      expect(result.prompt).toBe('a+ b c-');
    });

    it('should split group when selection starts before group and ends inside (increment)', () => {
      const prompt = 'a (b c)+';
      const result = adjustPromptAttention(prompt, 0, 4, 'increment');

      expect(result.prompt).toBe('(a b+ c)+');
    });

    it('should split group when selection starts before group and ends inside (decrement)', () => {
      const prompt = 'a (b c)+';
      const result = adjustPromptAttention(prompt, 0, 4, 'decrement');

      expect(result.prompt).toBe('a- b c+');
    });

    it('should handle nested groups with cross-boundary selection (increment)', () => {
      const prompt = '((a b)+)+ c';
      const result = adjustPromptAttention(prompt, 2, 11, 'increment');

      expect(result.prompt).toBe('((a b)++ c)+');
    });

    it('should handle nested groups with cross-boundary selection (decrement)', () => {
      const prompt = '((a b)+)+ c';
      const result = adjustPromptAttention(prompt, 2, 11, 'decrement');

      expect(result.prompt).toBe('(a b)+ c-');
    });

    it('should handle selection spanning multiple groups (increment)', () => {
      const prompt = '(a)+ (b)+';
      const result = adjustPromptAttention(prompt, 0, 9, 'increment');

      expect(result.prompt).toBe('(a b)++');
    });

    it('should handle selection spanning multiple groups (decrement)', () => {
      const prompt = '(a)+ (b)+';
      const result = adjustPromptAttention(prompt, 0, 9, 'decrement');

      expect(result.prompt).toBe('a b');
    });

    it('should split negative group correctly (decrement on negative group)', () => {
      const prompt = '(a b)- c';
      const result = adjustPromptAttention(prompt, 3, 8, 'decrement');

      expect(result.prompt).toBe('(a b- c)-');
    });

    it('should split negative group correctly (increment on negative group)', () => {
      const prompt = '(a b)- c';
      const result = adjustPromptAttention(prompt, 3, 8, 'increment');

      expect(result.prompt).toBe('a- b c+');
    });

    it('should handle multiple non-selected items in group', () => {
      const prompt = '(a b c)+ d';
      const result = adjustPromptAttention(prompt, 5, 10, 'decrement');

      expect(result.prompt).toBe('(a b)+ c d-');
    });

    it('should handle word with existing attention in group when crossing boundary', () => {
      const prompt = 'c (d- e)+';
      const result = adjustPromptAttention(prompt, 0, 5, 'increment');

      expect(result.prompt).toBe('c+ d e+');
    });

    it('should handle complex multi-group case', () => {
      const prompt = '(a+ b)+ c (d- e)+';
      const result = adjustPromptAttention(prompt, 8, 14, 'increment');

      expect(result.prompt).toBe('(a+ b c)+ d e+');
    });
  });

  describe('single word', () => {
    it('should add + when incrementing word without attention', () => {
      const prompt = 'hello world';
      const result = adjustPromptAttention(prompt, 0, 5, 'increment');

      expect(result.prompt).toBe('hello+ world');
    });

    it('should add - when decrementing word without attention', () => {
      const prompt = 'hello world';
      const result = adjustPromptAttention(prompt, 0, 5, 'decrement');

      expect(result.prompt).toBe('hello- world');
    });
  });

  describe('existing group', () => {
    it('should adjust group attention when cursor is at group boundary', () => {
      const prompt = '(hello world)+';
      const result = adjustPromptAttention(prompt, 13, 14, 'increment');

      expect(result.prompt).toBe('(hello world)++');
    });

    it('should remove group when attention becomes neutral', () => {
      const prompt = '(hello world)+';
      const result = adjustPromptAttention(prompt, 0, 14, 'decrement');

      expect(result.prompt).toBe('hello world');
    });
  });

  describe('multiple words without group', () => {
    it('should create new group with + when incrementing multiple words', () => {
      const prompt = 'hello world';
      const result = adjustPromptAttention(prompt, 0, 11, 'increment');

      expect(result.prompt).toBe('(hello world)+');
    });

    it('should create new group with - when decrementing multiple words', () => {
      const prompt = 'hello world';
      const result = adjustPromptAttention(prompt, 0, 11, 'decrement');

      expect(result.prompt).toBe('(hello world)-');
    });
  });

  describe('selection preservation', () => {
    it('should preserve selection when incrementing single word', () => {
      const prompt = 'hello world';
      const result = adjustPromptAttention(prompt, 0, 5, 'increment');
      expect(result.prompt).toBe('hello+ world');
      expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('hello+');
    });

    it('should preserve selection when incrementing group', () => {
      const prompt = '(hello world)+';
      const result = adjustPromptAttention(prompt, 0, 14, 'increment');
      expect(result.prompt).toBe('(hello world)++');
      expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('(hello world)++');
    });

    it('should preserve selection when splitting group', () => {
      const prompt = '(a b)+';
      const result = adjustPromptAttention(prompt, 1, 2, 'increment'); // Select 'a' (index 1 to 2)
      // 'a' becomes 1.21, 'b' stays 1.1
      // Result: (a+ b)+ which is equivalent to a++ b+
      expect(result.prompt).toBe('(a+ b)+');
      expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('a+');
    });
  });

  describe('numeric attention weights', () => {
    it('should preserve parentheses for single word with numeric weight when incrementing elsewhere', () => {
      const prompt = '(masterpiece)1.3, best quality';
      const len = prompt.length;
      // Select "best quality" and increment
      const bestQualityStart = prompt.indexOf('best quality');
      const result = adjustPromptAttention(prompt, bestQualityStart, len, 'increment');

      // masterpiece should keep its parens and exact weight
      expect(result.prompt).toContain('(masterpiece)1.3');
      expect(result.prompt).not.toContain('masterpiece1.3');
    });

    it('should not produce long floating point numbers for numeric weights', () => {
      const prompt = '(high detail)1.2, oil painting';
      const len = prompt.length;
      // Select "oil painting" and increment
      const oilStart = prompt.indexOf('oil painting');
      const result = adjustPromptAttention(prompt, oilStart, len, 'increment');

      // high detail should keep its exact weight, no floating point garbage
      expect(result.prompt).toContain('(high detail)1.2');
      expect(result.prompt).not.toMatch(/1\.19999/);
      expect(result.prompt).not.toMatch(/1\.20000/);
    });

    it('should preserve numeric weight 1.15 without floating point corruption', () => {
      const prompt = '(sunny midday light)1.15, landscape';
      const len = prompt.length;
      const landscapeStart = prompt.indexOf('landscape');
      const result = adjustPromptAttention(prompt, landscapeStart, len, 'increment');

      expect(result.prompt).toContain('(sunny midday light)1.15');
      expect(result.prompt).not.toMatch(/1\.15005/);
    });

    it('should normalize numeric 1.1 weight to + syntax', () => {
      const prompt = '(lush rolling hills)1.1, landscape';
      const len = prompt.length;
      const landscapeStart = prompt.indexOf('landscape');
      const result = adjustPromptAttention(prompt, landscapeStart, len, 'increment');

      // 1.1 is equivalent to +, normalization is acceptable
      expect(result.prompt).toMatch(/\(lush rolling hills\)(\+|1\.1)/);
    });

    it('should increment numeric weight correctly for single word', () => {
      const prompt = '(masterpiece)1.3';
      const result = adjustPromptAttention(prompt, 0, prompt.length, 'increment');

      // 1.3 + 0.1 = 1.4
      expect(result.prompt).toBe('(masterpiece)1.4');
    });

    it('should increment numeric weight correctly for multi-word group', () => {
      const prompt = '(high detail)1.2';
      const result = adjustPromptAttention(prompt, 0, prompt.length, 'increment');

      // 1.2 + 0.1 = 1.3
      expect(result.prompt).toBe('(high detail)1.3');
    });

    it('should decrement numeric weight correctly', () => {
      const prompt = '(masterpiece)1.3';
      const result = adjustPromptAttention(prompt, 0, prompt.length, 'decrement');

      // 1.3 - 0.1 = 1.2
      expect(result.prompt).toBe('(masterpiece)1.2');
    });

    it('should increment numeric weight 1.15 with additive step', () => {
      const prompt = '(sunny midday light)1.15';
      const result = adjustPromptAttention(prompt, 0, prompt.length, 'increment');

      // 1.15 + 0.1 = 1.25
      expect(result.prompt).toBe('(sunny midday light)1.25');
    });

    it('should decrement numeric weight 1.15 with additive step', () => {
      const prompt = '(sunny midday light)1.15';
      const result = adjustPromptAttention(prompt, 0, prompt.length, 'decrement');

      // 1.15 - 0.1 = 1.05
      expect(result.prompt).toBe('(sunny midday light)1.05');
    });

    it('should handle the full bug report prompt without corrupting non-selected weights', () => {
      const prompt =
        '(masterpiece)1.3, best quality, (high detail)1.2, oil painting, (sunny midday light)1.15, an old stone castle standing on a hill, medieval architecture, weathered stone walls, (lush rolling hills)1.1, expansive landscape, clear blue sky';
      const selStart = prompt.indexOf('clear blue sky');
      const selEnd = selStart + 'clear blue sky'.length;
      const result = adjustPromptAttention(prompt, selStart, selEnd, 'increment');

      // Non-selected numeric weights must be preserved exactly
      expect(result.prompt).toContain('(masterpiece)1.3');
      expect(result.prompt).toContain('(high detail)1.2');
      expect(result.prompt).toContain('(sunny midday light)1.15');
      // Selected text should be incremented
      expect(result.prompt).toContain('(clear blue sky)+');
      // No floating point garbage anywhere
      expect(result.prompt).not.toMatch(/\d\.\d{5,}/);
    });
  });
});
