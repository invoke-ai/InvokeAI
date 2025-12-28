import { describe, expect, it } from 'vitest';

import { parseTokens, serialize, tokenize } from './promptAST';

describe('promptAST', () => {
  describe('tokenize', () => {
    it('should tokenize basic text', () => {
      const tokens = tokenize('a cat');
      expect(tokens).toEqual([
        { type: 'word', value: 'a', start: 0, end: 1 },
        { type: 'whitespace', value: ' ', start: 1, end: 2 },
        { type: 'word', value: 'cat', start: 2, end: 5 },
      ]);
    });

    it('should tokenize groups with parentheses', () => {
      const tokens = tokenize('(a cat)');
      expect(tokens).toEqual([
        { type: 'lparen', start: 0, end: 1 },
        { type: 'word', value: 'a', start: 1, end: 2 },
        { type: 'whitespace', value: ' ', start: 2, end: 3 },
        { type: 'word', value: 'cat', start: 3, end: 6 },
        { type: 'rparen', start: 6, end: 7 },
      ]);
    });

    it('should tokenize escaped parentheses', () => {
      const tokens = tokenize('\\(medium\\)');
      expect(tokens).toEqual([
        { type: 'escaped_paren', value: '(', start: 0, end: 2 },
        { type: 'word', value: 'medium', start: 2, end: 8 },
        { type: 'escaped_paren', value: ')', start: 8, end: 10 },
      ]);
    });

    it('should tokenize mixed escaped and unescaped parentheses', () => {
      const tokens = tokenize('colored pencil \\(medium\\) (enhanced)');
      expect(tokens).toEqual([
        { type: 'word', value: 'colored', start: 0, end: 7 },
        { type: 'whitespace', value: ' ', start: 7, end: 8 },
        { type: 'word', value: 'pencil', start: 8, end: 14 },
        { type: 'whitespace', value: ' ', start: 14, end: 15 },
        { type: 'escaped_paren', value: '(', start: 15, end: 17 },
        { type: 'word', value: 'medium', start: 17, end: 23 },
        { type: 'escaped_paren', value: ')', start: 23, end: 25 },
        { type: 'whitespace', value: ' ', start: 25, end: 26 },
        { type: 'lparen', start: 26, end: 27 },
        { type: 'word', value: 'enhanced', start: 27, end: 35 },
        { type: 'rparen', start: 35, end: 36 },
      ]);
    });

    it('should tokenize groups with weights', () => {
      const tokens = tokenize('(a cat)1.2');
      expect(tokens).toEqual([
        { type: 'lparen', start: 0, end: 1 },
        { type: 'word', value: 'a', start: 1, end: 2 },
        { type: 'whitespace', value: ' ', start: 2, end: 3 },
        { type: 'word', value: 'cat', start: 3, end: 6 },
        { type: 'rparen', start: 6, end: 7 },
        { type: 'weight', value: 1.2, start: 7, end: 10 },
      ]);
    });

    it('should tokenize words with weights', () => {
      const tokens = tokenize('cat+');
      expect(tokens).toEqual([
        { type: 'word', value: 'cat', start: 0, end: 3 },
        { type: 'weight', value: '+', start: 3, end: 4 },
      ]);
    });

    it('should tokenize embeddings', () => {
      const tokens = tokenize('<embedding_name>');
      expect(tokens).toEqual([
        { type: 'lembed', start: 0, end: 1 },
        { type: 'word', value: 'embedding_name', start: 1, end: 15 },
        { type: 'rembed', start: 15, end: 16 },
      ]);
    });
  });

  describe('parseTokens', () => {
    it('should parse basic text', () => {
      const tokens = tokenize('a cat');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        { type: 'word', text: 'a', range: { start: 0, end: 1 }, attention: undefined },
        { type: 'whitespace', value: ' ', range: { start: 1, end: 2 } },
        { type: 'word', text: 'cat', range: { start: 2, end: 5 }, attention: undefined },
      ]);
    });

    it('should parse groups', () => {
      const tokens = tokenize('(a cat)');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        {
          type: 'group',
          range: { start: 0, end: 7 },
          attention: undefined,
          children: [
            { type: 'word', text: 'a', range: { start: 1, end: 2 }, attention: undefined },
            { type: 'whitespace', value: ' ', range: { start: 2, end: 3 } },
            { type: 'word', text: 'cat', range: { start: 3, end: 6 }, attention: undefined },
          ],
        },
      ]);
    });

    it('should parse escaped parentheses', () => {
      const tokens = tokenize('\\(medium\\)');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        { type: 'escaped_paren', value: '(', range: { start: 0, end: 2 } },
        { type: 'word', text: 'medium', range: { start: 2, end: 8 }, attention: undefined },
        { type: 'escaped_paren', value: ')', range: { start: 8, end: 10 } },
      ]);
    });

    it('should parse mixed escaped and unescaped parentheses', () => {
      const tokens = tokenize('colored pencil \\(medium\\) (enhanced)');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        { type: 'word', text: 'colored', range: { start: 0, end: 7 }, attention: undefined },
        { type: 'whitespace', value: ' ', range: { start: 7, end: 8 } },
        { type: 'word', text: 'pencil', range: { start: 8, end: 14 }, attention: undefined },
        { type: 'whitespace', value: ' ', range: { start: 14, end: 15 } },
        { type: 'escaped_paren', value: '(', range: { start: 15, end: 17 } },
        { type: 'word', text: 'medium', range: { start: 17, end: 23 }, attention: undefined },
        { type: 'escaped_paren', value: ')', range: { start: 23, end: 25 } },
        { type: 'whitespace', value: ' ', range: { start: 25, end: 26 } },
        {
          type: 'group',
          range: { start: 26, end: 36 },
          attention: undefined,
          children: [{ type: 'word', text: 'enhanced', range: { start: 27, end: 35 }, attention: undefined }],
        },
      ]);
    });

    it('should parse groups with attention', () => {
      const tokens = tokenize('(a cat)1.2');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        {
          type: 'group',
          attention: 1.2,
          range: { start: 0, end: 10 },
          children: [
            { type: 'word', text: 'a', range: { start: 1, end: 2 }, attention: undefined },
            { type: 'whitespace', value: ' ', range: { start: 2, end: 3 } },
            { type: 'word', text: 'cat', range: { start: 3, end: 6 }, attention: undefined },
          ],
        },
      ]);
    });

    it('should parse words with attention', () => {
      const tokens = tokenize('cat+');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([{ type: 'word', text: 'cat', attention: '+', range: { start: 0, end: 4 } }]);
    });

    it('should parse embeddings', () => {
      const tokens = tokenize('<embedding_name>');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([{ type: 'embedding', value: 'embedding_name', range: { start: 0, end: 16 } }]);
    });
  });

  describe('serialize', () => {
    it('should serialize basic text', () => {
      const tokens = tokenize('a cat');
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe('a cat');
    });

    it('should serialize groups', () => {
      const tokens = tokenize('(a cat)');
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe('(a cat)');
    });

    it('should serialize escaped parentheses', () => {
      const tokens = tokenize('\\(medium\\)');
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe('\\(medium\\)');
    });

    it('should serialize mixed escaped and unescaped parentheses', () => {
      const tokens = tokenize('colored pencil \\(medium\\) (enhanced)');
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe('colored pencil \\(medium\\) (enhanced)');
    });

    it('should serialize groups with attention', () => {
      const tokens = tokenize('(a cat)1.2');
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe('(a cat)1.2');
    });

    it('should serialize words with attention', () => {
      const tokens = tokenize('cat+');
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe('cat+');
    });

    it('should serialize embeddings', () => {
      const tokens = tokenize('<embedding_name>');
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe('<embedding_name>');
    });
  });

  describe('compel compatibility examples', () => {
    it('should handle escaped parentheses for literal text', () => {
      const prompt = 'A bear \\(with razor-sharp teeth\\) in a forest.';
      const tokens = tokenize(prompt);
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe(prompt);
    });

    it('should handle unescaped parentheses as grouping syntax', () => {
      const prompt = 'A bear (with razor-sharp teeth) in a forest.';
      const tokens = tokenize(prompt);
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe(prompt);
    });

    it('should handle colored pencil medium example', () => {
      const prompt = 'colored pencil \\(medium\\)';
      const tokens = tokenize(prompt);
      const ast = parseTokens(tokens);
      const result = serialize(ast);
      expect(result).toBe(prompt);
    });

    it('should distinguish between escaped and unescaped in same prompt', () => {
      const prompt = 'portrait \\(realistic\\) (high quality)1.2';
      const tokens = tokenize(prompt);
      const ast = parseTokens(tokens);

      // Should have escaped parens as nodes and a group with attention
      expect(ast).toEqual([
        { type: 'word', text: 'portrait', range: { start: 0, end: 8 }, attention: undefined },
        { type: 'whitespace', value: ' ', range: { start: 8, end: 9 } },
        { type: 'escaped_paren', value: '(', range: { start: 9, end: 11 } },
        { type: 'word', text: 'realistic', range: { start: 11, end: 20 }, attention: undefined },
        { type: 'escaped_paren', value: ')', range: { start: 20, end: 22 } },
        { type: 'whitespace', value: ' ', range: { start: 22, end: 23 } },
        {
          type: 'group',
          attention: 1.2,
          range: { start: 23, end: 40 },
          children: [
            { type: 'word', text: 'high', range: { start: 24, end: 28 }, attention: undefined },
            { type: 'whitespace', value: ' ', range: { start: 28, end: 29 } },
            { type: 'word', text: 'quality', range: { start: 29, end: 36 }, attention: undefined },
          ],
        },
      ]);

      const result = serialize(ast);
      expect(result).toBe(prompt);
    });
  });
});
