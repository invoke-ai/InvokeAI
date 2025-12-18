import { describe, expect, it } from 'vitest';

import { adjustPromptAttention } from './promptAttention';

describe('promptAttention', () => {
  describe('adjustPromptAttention', () => {
    describe('cross-boundary selection', () => {
      it('should wrap entire group and outside content when selection spans from inside group to outside', () => {
        // Bug case: "(a b)+ c" selecting "b through c" should become "((a b)+ c)+" not "(a (b)+"
        const prompt = '(a b)+ c';
        // Selection from 'b' (position 3) to end of 'c' (position 8)
        const result = adjustPromptAttention(prompt, 3, 8, 'increment');

        expect(result.prompt).toBe('((a b)+ c)+');
      });

      it('should wrap entire group and outside content when selection spans from inside group to outside (decrement)', () => {
        const prompt = '(a b)+ c';
        // Selection from 'b' (position 3) to end of 'c' (position 8)
        const result = adjustPromptAttention(prompt, 3, 8, 'decrement');

        expect(result.prompt).toBe('((a b)+ c)-');
      });

      it('should handle selection starting before group and ending inside', () => {
        const prompt = 'a (b c)+';
        // Selection from 'a' (position 0) to 'b' (position 3)
        const result = adjustPromptAttention(prompt, 0, 4, 'increment');

        expect(result.prompt).toBe('(a (b c)+)+');
      });

      it('should handle nested groups with cross-boundary selection', () => {
        const prompt = '((a b)+)+ c';
        // Selection from inside nested group to outside
        const result = adjustPromptAttention(prompt, 2, 11, 'increment');

        expect(result.prompt).toBe('(((a b)+)+ c)+');
      });

      it('should handle selection spanning multiple groups', () => {
        const prompt = '(a)+ (b)+';
        // Selection spanning both groups
        const result = adjustPromptAttention(prompt, 0, 9, 'increment');

        expect(result.prompt).toBe('((a)+ (b)+)+');
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
        // Cursor at the closing paren
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
  });
});
