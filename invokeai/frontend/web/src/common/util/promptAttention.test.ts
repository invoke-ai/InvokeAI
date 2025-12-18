import { describe, expect, it } from 'vitest';

import { adjustPromptAttention } from './promptAttention';

describe('promptAttention', () => {
  describe('adjustPromptAttention', () => {
    describe('cross-boundary selection', () => {
      it('should split group and apply attention when selection spans from inside group to outside (increment)', () => {
        // "(a b)+ c" selecting "b through c":
        // - a was in + group, not selected → gets + attention → a+
        // - b was in + group (effective +), selected, increment → ++ → b++
        // - c was neutral, selected, increment → +
        const prompt = '(a b)+ c';
        // Selection from 'b' (position 3) to end of 'c' (position 8)
        const result = adjustPromptAttention(prompt, 3, 8, 'increment');

        expect(result.prompt).toBe('a+ b++ c+');
      });

      it('should split group and apply attention when selection spans from inside group to outside (decrement)', () => {
        // "(a b)+ c" selecting "b through c":
        // - a was in + group, not selected → gets + attention → a+
        // - b was in + group (effective +), selected, decrement → neutralizes → b
        // - c was neutral, selected, decrement → -
        const prompt = '(a b)+ c';
        // Selection from 'b' (position 3) to end of 'c' (position 8)
        const result = adjustPromptAttention(prompt, 3, 8, 'decrement');

        expect(result.prompt).toBe('a+ b c-');
      });

      it('should split group when selection starts before group and ends inside (increment)', () => {
        // "a (b c)+" selecting "a and b":
        // - a was neutral, selected, increment → +
        // - b was in + group (effective +), selected, increment → ++
        // - c was in + group, not selected → keeps + attention
        const prompt = 'a (b c)+';
        // Selection from 'a' (position 0) to 'b' (position 3)
        const result = adjustPromptAttention(prompt, 0, 4, 'increment');

        expect(result.prompt).toBe('a+ b++ c+');
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

      it('should split negative group correctly (decrement on negative group)', () => {
        // "(a b)- c" selecting "b through c":
        // - a was in - group, not selected → keeps - attention → a-
        // - b was in - group (effective -), selected, decrement → -- → b--
        // - c was neutral, selected, decrement → -
        const prompt = '(a b)- c';
        const result = adjustPromptAttention(prompt, 3, 8, 'decrement');

        expect(result.prompt).toBe('a- b-- c-');
      });

      it('should split negative group correctly (increment on negative group)', () => {
        // "(a b)- c" selecting "b through c":
        // - a was in - group, not selected → keeps - attention → a-
        // - b was in - group (effective -), selected, increment → neutralizes → b
        // - c was neutral, selected, increment → +
        const prompt = '(a b)- c';
        const result = adjustPromptAttention(prompt, 3, 8, 'increment');

        expect(result.prompt).toBe('a- b c+');
      });

      it('should handle multiple non-selected items in group', () => {
        // "(a b c)+ d" selecting only "c d":
        // - a, b not selected → keep + attention
        // - c was in + group (effective +), selected, decrement → neutralizes → c
        // - d selected, decrement → -
        const prompt = '(a b c)+ d';
        const result = adjustPromptAttention(prompt, 5, 10, 'decrement');

        expect(result.prompt).toBe('a+ b+ c d-');
      });

      it('should handle word with existing attention in group when crossing boundary', () => {
        // "x (d- e)+" with "x d" selected and incremented:
        // - x at root, selected → increment → x+
        // - d HAS own attention (-), selected → adjust own only (- → neutral) → d
        // - e not selected → gets group's + attention → e+
        const prompt = 'x (d- e)+';
        // Select from x to d (positions 0 to 5, covering x and d-)
        const result = adjustPromptAttention(prompt, 0, 5, 'increment');

        expect(result.prompt).toBe('x+ d e+');
      });

      it('should handle complex multi-group case', () => {
        // "(a+ b)+ c (d- e)+" with "c d" selected and incremented:
        // - First group untouched since no children selected
        // - c at root, selected → increment → c+
        // - Second group split:
        //   - d HAS own attention (-), selected → adjust own only (- → neutral) → d
        //   - e not selected → gets group's + attention → e+
        const prompt = '(a+ b)+ c (d- e)+';
        // Select from c to d
        const result = adjustPromptAttention(prompt, 8, 14, 'increment');

        expect(result.prompt).toBe('(a+ b)+ c+ d e+');
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
