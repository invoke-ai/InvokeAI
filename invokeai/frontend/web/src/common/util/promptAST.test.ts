import { describe, expect, it } from 'vitest';

import { parseTokens, serialize, tokenize } from './promptAST';

describe('promptAST', () => {
  describe('tokenize', () => {
    it('should tokenize basic text', () => {
      const tokens = tokenize('a cat');
      expect(tokens).toEqual([
        { type: 'word', value: 'a' },
        { type: 'whitespace', value: ' ' },
        { type: 'word', value: 'cat' },
      ]);
    });

    it('should tokenize groups with parentheses', () => {
      const tokens = tokenize('(a cat)');
      expect(tokens).toEqual([
        { type: 'lparen' },
        { type: 'word', value: 'a' },
        { type: 'whitespace', value: ' ' },
        { type: 'word', value: 'cat' },
        { type: 'rparen' },
      ]);
    });

    it('should tokenize escaped parentheses', () => {
      const tokens = tokenize('\\(medium\\)');
      expect(tokens).toEqual([
        { type: 'escaped_paren', value: '(' },
        { type: 'word', value: 'medium' },
        { type: 'escaped_paren', value: ')' },
      ]);
    });

    it('should tokenize mixed escaped and unescaped parentheses', () => {
      const tokens = tokenize('colored pencil \\(medium\\) (enhanced)');
      expect(tokens).toEqual([
        { type: 'word', value: 'colored' },
        { type: 'whitespace', value: ' ' },
        { type: 'word', value: 'pencil' },
        { type: 'whitespace', value: ' ' },
        { type: 'escaped_paren', value: '(' },
        { type: 'word', value: 'medium' },
        { type: 'escaped_paren', value: ')' },
        { type: 'whitespace', value: ' ' },
        { type: 'lparen' },
        { type: 'word', value: 'enhanced' },
        { type: 'rparen' },
      ]);
    });

    it('should tokenize groups with weights', () => {
      const tokens = tokenize('(a cat)1.2');
      expect(tokens).toEqual([
        { type: 'lparen' },
        { type: 'word', value: 'a' },
        { type: 'whitespace', value: ' ' },
        { type: 'word', value: 'cat' },
        { type: 'rparen' },
        { type: 'weight', value: 1.2 },
      ]);
    });

    it('should tokenize words with weights', () => {
      const tokens = tokenize('cat+');
      expect(tokens).toEqual([
        { type: 'word', value: 'cat' },
        { type: 'weight', value: '+' },
      ]);
    });

    it('should tokenize embeddings', () => {
      const tokens = tokenize('<embedding_name>');
      expect(tokens).toEqual([{ type: 'lembed' }, { type: 'word', value: 'embedding_name' }, { type: 'rembed' }]);
    });
  });

  describe('parseTokens', () => {
    it('should parse basic text', () => {
      const tokens = tokenize('a cat');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        { type: 'word', text: 'a' },
        { type: 'whitespace', value: ' ' },
        { type: 'word', text: 'cat' },
      ]);
    });

    it('should parse groups', () => {
      const tokens = tokenize('(a cat)');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        {
          type: 'group',
          children: [
            { type: 'word', text: 'a' },
            { type: 'whitespace', value: ' ' },
            { type: 'word', text: 'cat' },
          ],
        },
      ]);
    });

    it('should parse escaped parentheses', () => {
      const tokens = tokenize('\\(medium\\)');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        { type: 'escaped_paren', value: '(' },
        { type: 'word', text: 'medium' },
        { type: 'escaped_paren', value: ')' },
      ]);
    });

    it('should parse mixed escaped and unescaped parentheses', () => {
      const tokens = tokenize('colored pencil \\(medium\\) (enhanced)');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([
        { type: 'word', text: 'colored' },
        { type: 'whitespace', value: ' ' },
        { type: 'word', text: 'pencil' },
        { type: 'whitespace', value: ' ' },
        { type: 'escaped_paren', value: '(' },
        { type: 'word', text: 'medium' },
        { type: 'escaped_paren', value: ')' },
        { type: 'whitespace', value: ' ' },
        {
          type: 'group',
          children: [{ type: 'word', text: 'enhanced' }],
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
          children: [
            { type: 'word', text: 'a' },
            { type: 'whitespace', value: ' ' },
            { type: 'word', text: 'cat' },
          ],
        },
      ]);
    });

    it('should parse words with attention', () => {
      const tokens = tokenize('cat+');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([{ type: 'word', text: 'cat', attention: '+' }]);
    });

    it('should parse embeddings', () => {
      const tokens = tokenize('<embedding_name>');
      const ast = parseTokens(tokens);
      expect(ast).toEqual([{ type: 'embedding', value: 'embedding_name' }]);
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
        { type: 'word', text: 'portrait' },
        { type: 'whitespace', value: ' ' },
        { type: 'escaped_paren', value: '(' },
        { type: 'word', text: 'realistic' },
        { type: 'escaped_paren', value: ')' },
        { type: 'whitespace', value: ' ' },
        {
          type: 'group',
          attention: 1.2,
          children: [
            { type: 'word', text: 'high' },
            { type: 'whitespace', value: ' ' },
            { type: 'word', text: 'quality' },
          ],
        },
      ]);

      const result = serialize(ast);
      expect(result).toBe(prompt);
    });
  });
});
