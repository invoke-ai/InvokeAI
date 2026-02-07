import { describe, expect, it } from 'vitest';

import { adjustPromptAttention } from './promptAttention';

/**
 * Helper: select by substring match within the prompt.
 * If `selected` is a string, finds it in the prompt and uses its position.
 * If `selected` is a [start, end] tuple, uses those positions directly.
 */
function adj(
  prompt: string,
  selected: string | [number, number],
  direction: 'increment' | 'decrement',
  prefersNumericWeights = false
) {
  const [start, end] =
    typeof selected === 'string' ? [prompt.indexOf(selected), prompt.indexOf(selected) + selected.length] : selected;
  return adjustPromptAttention(prompt, start, end, direction, prefersNumericWeights);
}

/** Helper that calls adj with prefersNumericWeights=true */
function adjNumeric(prompt: string, selected: string | [number, number], direction: 'increment' | 'decrement') {
  return adj(prompt, selected, direction, true);
}

describe('adjustPromptAttention', () => {
  // Basic Attention

  describe('single word', () => {
    it.each([
      ['hello world', 'hello', 'increment', 'hello+ world'],
      ['hello world', 'hello', 'decrement', 'hello- world'],
      ['hello+ world', 'hello+', 'increment', 'hello++ world'],
      ['hello+ world', 'hello+', 'decrement', 'hello world'],
      ['hello- world', 'hello-', 'decrement', 'hello-- world'],
      ['hello- world', 'hello-', 'increment', 'hello world'],
    ] as const)('%s [%s] %s → %s', (prompt, selected, direction, expected) => {
      expect(adj(prompt, selected, direction).prompt).toBe(expected);
    });
  });

  describe('multiple words', () => {
    it.each([
      ['hello world', [0, 11] as [number, number], 'increment', '(hello world)+'],
      ['hello world', [0, 11] as [number, number], 'decrement', '(hello world)-'],
    ] as const)('%s [%s] %s → %s', (prompt, selected, direction, expected) => {
      expect(adj(prompt, selected, direction).prompt).toBe(expected);
    });
  });

  // Existing Groups

  describe('existing groups', () => {
    it('should increment group when cursor is at group boundary', () => {
      expect(adj('(hello world)+', [13, 14], 'increment').prompt).toBe('(hello world)++');
    });

    it('should remove group when attention becomes neutral', () => {
      expect(adj('(hello world)+', [0, 14], 'decrement').prompt).toBe('hello world');
    });

    it('should increment inner word within group', () => {
      const result = adj('(a b)+', [1, 2], 'increment');
      expect(result.prompt).toBe('(a+ b)+');
    });
  });

  // Cross-Boundary Selection

  describe('cross-boundary selection', () => {
    it.each([
      // Selection from inside group to outside
      ['(a b)+ c', [3, 8], 'increment', '(a b+ c)+'],
      ['(a b)+ c', [3, 8], 'decrement', 'a+ b c-'],
      // Selection from outside to inside group
      ['a (b c)+', [0, 4], 'increment', '(a b+ c)+'],
      ['a (b c)+', [0, 4], 'decrement', 'a- b c+'],
      // Nested groups
      ['((a b)+)+ c', [2, 11], 'increment', '((a b)++ c)+'],
      ['((a b)+)+ c', [2, 11], 'decrement', '(a b)+ c-'],
      // Spanning multiple groups
      ['(a)+ (b)+', [0, 9], 'increment', '(a b)++'],
      ['(a)+ (b)+', [0, 9], 'decrement', 'a b'],
      // Negative groups
      ['(a b)- c', [3, 8], 'decrement', '(a b- c)-'],
      ['(a b)- c', [3, 8], 'increment', 'a- b c+'],
      // Multiple non-selected items in group
      ['(a b c)+ d', [5, 10], 'decrement', '(a b)+ c d-'],
      // Word with existing attention crossing boundary
      ['c (d- e)+', [0, 5], 'increment', 'c+ d e+'],
      // Complex multi-group
      ['(a+ b)+ c (d- e)+', [8, 14], 'increment', '(a+ b c)+ d e+'],
    ] as const)('%s [%s] %s → %s', (prompt, selected, direction, expected) => {
      expect(adj(prompt, selected as string | [number, number], direction).prompt).toBe(expected);
    });
  });

  // Selection Preservation

  describe('selection preservation', () => {
    it('should track selection when incrementing single word', () => {
      const result = adj('hello world', 'hello', 'increment');
      expect(result.prompt).toBe('hello+ world');
      expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('hello+');
    });

    it('should track selection when incrementing full group', () => {
      const result = adj('(hello world)+', [0, 14], 'increment');
      expect(result.prompt).toBe('(hello world)++');
      expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('(hello world)++');
    });

    it('should track selection when splitting group', () => {
      const result = adj('(a b)+', [1, 2], 'increment');
      expect(result.prompt).toBe('(a+ b)+');
      expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('a+');
    });
  });

  // Numeric Attention Weights

  describe('numeric attention weights', () => {
    it.each([
      // Increment / decrement numeric weights with additive step
      ['(masterpiece)1.3', [0, 16], 'increment', '(masterpiece)1.4'],
      ['(masterpiece)1.3', [0, 16], 'decrement', '(masterpiece)1.2'],
      ['(high detail)1.2', [0, 16], 'increment', '(high detail)1.3'],
      ['(sunny midday light)1.15', [0, 24], 'increment', '(sunny midday light)1.25'],
      ['(sunny midday light)1.15', [0, 24], 'decrement', '(sunny midday light)1.05'],
    ] as const)('%s [%s] %s → %s', (prompt, selected, direction, expected) => {
      expect(adj(prompt, selected as [number, number], direction).prompt).toBe(expected);
    });

    it('should preserve non-selected numeric weights when adjusting elsewhere', () => {
      const prompt = '(masterpiece)1.3, best quality';
      const result = adj(prompt, 'best quality', 'increment');
      expect(result.prompt).toContain('(masterpiece)1.3');
      expect(result.prompt).not.toContain('masterpiece1.3');
    });

    it('should not produce floating point garbage', () => {
      const prompt = '(high detail)1.2, oil painting';
      const result = adj(prompt, 'oil painting', 'increment');
      expect(result.prompt).toContain('(high detail)1.2');
      expect(result.prompt).not.toMatch(/1\.19999/);
      expect(result.prompt).not.toMatch(/1\.20000/);
    });

    it('should preserve numeric weight 1.15 without corruption', () => {
      const prompt = '(sunny midday light)1.15, landscape';
      const result = adj(prompt, 'landscape', 'increment');
      expect(result.prompt).toContain('(sunny midday light)1.15');
      expect(result.prompt).not.toMatch(/1\.15005/);
    });

    it('should normalize numeric 1.1 weight to + syntax', () => {
      const prompt = '(lush rolling hills)1.1, landscape';
      const result = adj(prompt, 'landscape', 'increment');
      expect(result.prompt).toMatch(/\(lush rolling hills\)(\+|1\.1)/);
    });

    it('should handle the full complex prompt without corrupting non-selected weights', () => {
      const prompt =
        '(masterpiece)1.3, best quality, (high detail)1.2, oil painting, (sunny midday light)1.15, an old stone castle standing on a hill, medieval architecture, weathered stone walls, (lush rolling hills)1.1, expansive landscape, clear blue sky';
      const result = adj(prompt, 'clear blue sky', 'increment');

      expect(result.prompt).toContain('(masterpiece)1.3');
      expect(result.prompt).toContain('(high detail)1.2');
      expect(result.prompt).toContain('(sunny midday light)1.15');
      expect(result.prompt).toContain('(clear blue sky)+');
      expect(result.prompt).not.toMatch(/\d\.\d{5,}/);
    });
  });

  // Prompt Functions

  describe('prompt functions', () => {
    describe('within a single argument', () => {
      it.each([
        // Single word inside an arg
        ["('hello world', 'other').and()", 'hello', 'increment', "('hello+ world', 'other').and()"],
        ["('hello world', 'other').and()", 'hello', 'decrement', "('hello- world', 'other').and()"],
        // Multiple words in second arg
        ["('a', 'hello world').or()", 'hello world', 'increment', "('a', '(hello world)+').or()"],
        ["('a', 'hello world').or()", 'hello world', 'decrement', "('a', '(hello world)-').or()"],
        // Single word in .blend()
        ["('one two', 'three four').blend(0.7, 0.3)", 'two', 'increment', "('one two+', 'three four').blend(0.7, 0.3)"],
      ] as const)('%s [%s] %s → %s', (prompt, selected, direction, expected) => {
        expect(adj(prompt, selected, direction).prompt).toBe(expected);
      });
    });

    describe('across argument separator', () => {
      it('should adjust both args simultaneously when selection spans separator (increment)', () => {
        const prompt = "('one two', 'three four').and()";
        // Select across the separator: "two', 'three"
        const start = prompt.indexOf('two');
        const end = prompt.indexOf('three') + 'three'.length;
        const result = adjustPromptAttention(prompt, start, end, 'increment');
        expect(result.prompt).toBe("('one two+', 'three+ four').and()");
      });

      it('should adjust both args simultaneously when selection spans separator (decrement)', () => {
        const prompt = "('one two', 'three four').and()";
        const start = prompt.indexOf('two');
        const end = prompt.indexOf('three') + 'three'.length;
        const result = adjustPromptAttention(prompt, start, end, 'decrement');
        expect(result.prompt).toBe("('one two-', 'three- four').and()");
      });

      it('should adjust across separator for .or()', () => {
        const prompt = "('alpha beta', 'gamma delta').or()";
        const start = prompt.indexOf('beta');
        const end = prompt.indexOf('gamma') + 'gamma'.length;
        const result = adjustPromptAttention(prompt, start, end, 'increment');
        expect(result.prompt).toBe("('alpha beta+', 'gamma+ delta').or()");
      });

      it('should adjust across separator for .blend() preserving params', () => {
        const prompt = "('one two', 'three four').blend(0.7, 0.3)";
        const start = prompt.indexOf('two');
        const end = prompt.indexOf('three') + 'three'.length;
        const result = adjustPromptAttention(prompt, start, end, 'increment');
        expect(result.prompt).toBe("('one two+', 'three+ four').blend(0.7, 0.3)");
      });

      it('should handle repeated increment across separator', () => {
        const prompt = "('one two+', 'three+ four').and()";
        const start = prompt.indexOf('two');
        const end = prompt.indexOf('three') + 'three'.length;
        // "two+" is at the boundary, "three+" is at the boundary
        const result = adjustPromptAttention(prompt, start, end, 'increment');
        expect(result.prompt).toBe("('one two++', 'three++ four').and()");
      });
    });

    describe('whole function selected', () => {
      it('should increment all content in all args when whole function is selected', () => {
        const prompt = "('one', 'two').and()";
        const result = adjustPromptAttention(prompt, 0, prompt.length, 'increment');
        expect(result.prompt).toBe("('one+', 'two+').and()");
      });

      it('should decrement all content in all args', () => {
        const prompt = "('one', 'two').and()";
        const result = adjustPromptAttention(prompt, 0, prompt.length, 'decrement');
        expect(result.prompt).toBe("('one-', 'two-').and()");
      });

      it('should increment all args of .blend() preserving params', () => {
        const prompt = "('one', 'two').blend(0.7, 0.3)";
        const result = adjustPromptAttention(prompt, 0, prompt.length, 'increment');
        expect(result.prompt).toBe("('one+', 'two+').blend(0.7, 0.3)");
      });
    });

    describe('prompt function embedded in larger prompt', () => {
      it('should adjust only the targeted region outside the function', () => {
        const prompt = "some text, ('a', 'b').and(), more text";
        const result = adj(prompt, 'some', 'increment');
        expect(result.prompt).toContain('some+');
        expect(result.prompt).toContain("('a', 'b').and()");
      });

      it('should adjust only the targeted region inside the function', () => {
        const prompt = "prefix ('alpha beta', 'gamma').and() suffix";
        const result = adj(prompt, 'alpha', 'increment');
        expect(result.prompt).toContain("'alpha+ beta'");
        expect(result.prompt).toContain('prefix');
        expect(result.prompt).toContain('suffix');
      });

      it('should adjust text outside and inside function when selection spans boundary', () => {
        const prompt = "text ('one two', 'three').and()";
        // Select from 'text' through 'one'
        const start = prompt.indexOf('text');
        const end = prompt.indexOf('one') + 'one'.length;
        const result = adjustPromptAttention(prompt, start, end, 'increment');
        expect(result.prompt).toContain('text+');
        expect(result.prompt).toContain("'one+ two'");
      });
    });

    describe('prompt function with existing attention inside args', () => {
      it('should further increment already-weighted word inside arg', () => {
        const prompt = "('hello+', 'world').and()";
        // Select hello+ (the word with its weight marker)
        const result = adj(prompt, 'hello+', 'increment');
        expect(result.prompt).toBe("('hello++', 'world').and()");
      });

      it('should cancel attention to neutral inside arg', () => {
        const prompt = "('hello+', 'world').and()";
        const result = adj(prompt, 'hello+', 'decrement');
        expect(result.prompt).toBe("('hello', 'world').and()");
      });

      it('should handle group attention inside arg', () => {
        const prompt = "('(a b)+', 'c').and()";
        // Select everything in first arg
        const start = prompt.indexOf('(a b)+');
        const end = start + '(a b)+'.length;
        const result = adjustPromptAttention(prompt, start, end, 'increment');
        expect(result.prompt).toBe("('(a b)++', 'c').and()");
      });
    });

    describe('three-arg prompt functions', () => {
      it('should adjust a word in one arg of a three-arg blend', () => {
        const prompt = "('a', 'b', 'c').blend(0.5, 0.3, 0.2)";
        const result = adj(prompt, 'b', 'increment');
        expect(result.prompt).toBe("('a', 'b+', 'c').blend(0.5, 0.3, 0.2)");
      });

      it('should adjust across two separators in a three-arg blend', () => {
        const prompt = "('aa bb', 'cc dd', 'ee ff').blend(0.5, 0.3, 0.2)";
        // Select from bb through ee
        const start = prompt.indexOf('bb');
        const end = prompt.indexOf('ee') + 'ee'.length;
        const result = adjustPromptAttention(prompt, start, end, 'increment');
        expect(result.prompt).toBe("('aa bb+', '(cc dd)+', 'ee+ ff').blend(0.5, 0.3, 0.2)");
      });
    });
  });

  // Selection Preservation with Prompt Functions

  describe('selection preservation with prompt functions', () => {
    it('should track selection for single word inside prompt function arg', () => {
      const prompt = "('hello world', 'other').and()";
      const result = adj(prompt, 'hello', 'increment');
      expect(result.prompt).toBe("('hello+ world', 'other').and()");
      expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('hello+');
    });

    it('should track selection spanning across prompt function separator', () => {
      const prompt = "('one two', 'three four').and()";
      const start = prompt.indexOf('two');
      const end = prompt.indexOf('three') + 'three'.length;
      const result = adjustPromptAttention(prompt, start, end, 'increment');
      expect(result.prompt).toBe("('one two+', 'three+ four').and()");
      // Selection should span from 'two+' through 'three+' (including structural chars between)
      const sel = result.prompt.slice(result.selectionStart, result.selectionEnd);
      expect(sel).toContain('two+');
      expect(sel).toContain('three+');
    });
  });

  // Edge Cases

  describe('edge cases', () => {
    it('should return prompt unchanged when no selection overlap', () => {
      const prompt = 'hello world';
      const result = adjustPromptAttention(prompt, 5, 5, 'increment');
      // Cursor at the boundary between hello and space — should still find a terminal
      expect(result.prompt).toBeDefined();
    });

    it('should handle empty prompt', () => {
      const result = adjustPromptAttention('', 0, 0, 'increment');
      expect(result.prompt).toBe('');
    });

    it('should not modify prompt function structure when cursor is on structural char', () => {
      const prompt = "('a', 'b').and()";
      // Cursor on the dot between ) and and
      const dotPos = prompt.indexOf('.and');
      const result = adjustPromptAttention(prompt, dotPos, dotPos, 'increment');
      // Should either not change or only affect content, not break the structure
      expect(result.prompt).toContain('.and()');
    });
  });

  // Numeric Weight Preference

  describe('prefersNumericWeights', () => {
    describe('single word (no existing attention)', () => {
      it.each([
        ['hello world', 'hello', 'increment', '(hello)1.1 world'],
        ['hello world', 'hello', 'decrement', '(hello)0.9 world'],
        ['hello world', 'world', 'increment', 'hello (world)1.1'],
        ['hello world', 'world', 'decrement', 'hello (world)0.9'],
      ] as const)('%s [%s] %s → %s', (prompt, selected, direction, expected) => {
        expect(adjNumeric(prompt, selected, direction).prompt).toBe(expected);
      });
    });

    describe('successive numeric adjustments', () => {
      it('should use additive step on second increment', () => {
        const result = adjNumeric('(hello)1.1 world', '(hello)1.1', 'increment');
        expect(result.prompt).toBe('(hello)1.2 world');
      });

      it('should use additive step on second decrement', () => {
        const result = adjNumeric('(hello)0.9 world', '(hello)0.9', 'decrement');
        expect(result.prompt).toBe('(hello)0.8 world');
      });

      it('should return to neutral from 1.1 on decrement', () => {
        const result = adjNumeric('(hello)1.1 world', '(hello)1.1', 'decrement');
        expect(result.prompt).toBe('hello world');
      });
    });

    describe('does not convert existing +/- attention on unselected terminals', () => {
      it('should preserve +/- on unselected word when adjusting another', () => {
        const result = adjNumeric('hello+ world', 'world', 'increment');
        expect(result.prompt).toContain('hello+');
        expect(result.prompt).toContain('(world)1.1');
      });

      it('should preserve - on unselected word', () => {
        const result = adjNumeric('hello- world', 'world', 'decrement');
        expect(result.prompt).toContain('hello-');
        expect(result.prompt).toContain('(world)0.9');
      });
    });

    describe('existing +/- attention on selected terminals', () => {
      it('should increment existing + word with multiplicative step (respects existing style)', () => {
        const result = adjNumeric('hello+ world', 'hello+', 'increment');
        // The terminal already has explicit +/- attention, so it keeps that style
        expect(result.prompt).toBe('hello++ world');
      });

      it('should decrement existing + word to neutral', () => {
        const result = adjNumeric('hello+ world', 'hello+', 'decrement');
        expect(result.prompt).toBe('hello world');
      });
    });

    describe('existing numeric attention on selected terminals', () => {
      it('should increment existing numeric weight additively', () => {
        const result = adjNumeric('(detail)1.3 world', '(detail)1.3', 'increment');
        expect(result.prompt).toBe('(detail)1.4 world');
      });

      it('should decrement existing numeric weight additively', () => {
        const result = adjNumeric('(detail)1.3 world', '(detail)1.3', 'decrement');
        expect(result.prompt).toBe('(detail)1.2 world');
      });
    });

    describe('multiple words selected', () => {
      it('should wrap multiple words in numeric group on increment', () => {
        const result = adjNumeric('hello world', [0, 11], 'increment');
        expect(result.prompt).toBe('(hello world)1.1');
      });

      it('should wrap multiple words in numeric group on decrement', () => {
        const result = adjNumeric('hello world', [0, 11], 'decrement');
        expect(result.prompt).toBe('(hello world)0.9');
      });
    });

    describe('inside prompt functions', () => {
      it('should use numeric format inside prompt function arg', () => {
        const prompt = "('hello world', 'other').and()";
        const result = adjNumeric(prompt, 'hello', 'increment');
        expect(result.prompt).toBe("('(hello)1.1 world', 'other').and()");
      });

      it('should use numeric format across prompt function separator', () => {
        const prompt = "('one two', 'three four').and()";
        const start = prompt.indexOf('two');
        const end = prompt.indexOf('three') + 'three'.length;
        const result = adjustPromptAttention(prompt, start, end, 'increment', true);
        expect(result.prompt).toBe("('one (two)1.1', '(three)1.1 four').and()");
      });
    });

    describe('without prefersNumericWeights (default behavior unchanged)', () => {
      it('should still use +/- syntax by default', () => {
        expect(adj('hello world', 'hello', 'increment').prompt).toBe('hello+ world');
        expect(adj('hello world', 'hello', 'decrement').prompt).toBe('hello- world');
      });

      it('should still use +/- for multiple words by default', () => {
        expect(adj('hello world', [0, 11], 'increment').prompt).toBe('(hello world)+');
      });
    });
  });
});
