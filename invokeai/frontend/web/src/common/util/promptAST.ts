/**
 * Expected as either '+', '-', '++', '--', etc. or a numeric string like '1.2', '0.8', etc.
 */
export type Attention = string | number;

type Word = string;

type Punct = string;

type Whitespace = string;

type Embedding = string;

export type Token =
  | { type: 'word'; value: Word }
  | { type: 'whitespace'; value: Whitespace }
  | { type: 'punct'; value: Punct }
  | { type: 'lparen' }
  | { type: 'rparen' }
  | { type: 'weight'; value: Attention }
  | { type: 'lembed' }
  | { type: 'rembed' };

export type ASTNode =
  | { type: 'word'; text: Word; attention?: Attention }
  | { type: 'group'; children: ASTNode[]; attention?: Attention }
  | { type: 'embedding'; value: Embedding }
  | { type: 'whitespace'; value: Whitespace }
  | { type: 'punct'; value: Punct };

/**
 * Convert a prompt string into an AST.
 * @param prompt string
 * @returns ASTNode[]
 */
export function tokenize(prompt: string): Token[] {
  if (!prompt) {
    return [];
  }

  let i = 0;
  let tokens: Token[] = [];

  while (i < prompt.length) {
    const char = prompt[i];
    if (!char) {
      break;
    }

    // Whitespace (including newlines)
    if (/\s/.test(char)) {
      tokens.push({ type: 'whitespace', value: char });
      i++;
      continue;
    }

    // Parentheses
    if (char === '(') {
      tokens.push({ type: 'lparen' });
      i++;
      continue;
    }

    if (char === ')') {
      // Look ahead for weight like ')1.1' or ')-0.9' or ')+' or ')-'
      const weightMatch = prompt.slice(i + 1).match(/^[+-]?(\d+(\.\d+)?|[+-]+)/);
      if (weightMatch && weightMatch[0]) {
        let weight: Attention = weightMatch[0];
        if (!isNaN(Number(weight))) {
          weight = Number(weight);
        }
        tokens.push({ type: 'rparen' });
        tokens.push({ type: 'weight', value: weight });
        i += 1 + weightMatch[0].length;
        continue;
      }
      tokens.push({ type: 'rparen' });
      i++;
      continue;
    }

    // Handle punctuation (comma, period, etc.)
    if (/[,.]/.test(char)) {
      tokens.push({ type: 'punct', value: char });
      i++;
      continue;
    }

    // Read a word (letters, digits, underscores)
    if (/[a-zA-Z0-9_]/.test(char)) {
      let j = i;
      while (j < prompt.length && /[a-zA-Z0-9_]/.test(prompt[j]!)) {
        j++;
      }
      const word = prompt.slice(i, j);
      tokens.push({ type: 'word', value: word });

      // Check for weight immediately after word (e.g., "Lorem+", "consectetur-")
      const weightMatch = prompt.slice(j).match(/^[+-]?(\d+(\.\d+)?|[+-]+)/);
      if (weightMatch && weightMatch[0]) {
        tokens.push({ type: 'weight', value: weightMatch[0] });
        i = j + weightMatch[0].length;
      } else {
        i = j;
      }
      continue;
    }

    // Embeddings
    if (char === '<') {
      tokens.push({ type: 'lembed' });
      i++;
      continue;
    }

    if (char === '>') {
      tokens.push({ type: 'rembed' });
      i++;
      continue;
    }

    // Any other single character punctuation
    if (!/\s/.test(char)) {
      tokens.push({ type: 'punct', value: char });
    }

    i++;
  }

  return tokens;
}

/**
 * Convert tokens into an AST.
 * @param tokens Token[]
 * @returns ASTNode[]
 */
export function parseTokens(tokens: Token[]): ASTNode[] {
  let pos = 0;

  function peek(): Token | undefined {
    return tokens[pos];
  }

  function consume(): Token | undefined {
    return tokens[pos++];
  }

  function parseGroup(): ASTNode[] {
    const nodes: ASTNode[] = [];

    while (pos < tokens.length) {
      const token = peek();
      if (!token || token.type === 'rparen') {
        break;
      }
      // console.log('Parsing token:', token);

      switch (token.type) {
        case 'whitespace': {
          const wsToken = consume() as Token & { type: 'whitespace' };
          nodes.push({ type: 'whitespace', value: wsToken.value });
          break;
        }
        case 'lparen': {
          consume();
          const groupChildren = parseGroup();

          let attention: Attention | undefined;
          if (peek()?.type === 'rparen') {
            consume(); // consume ')'
            if (peek()?.type === 'weight') {
              attention = (consume() as Token & { type: 'weight' }).value;
            }
          }

          nodes.push({ type: 'group', children: groupChildren, attention });
          break;
        }
        case 'lembed': {
          consume(); // consume '<'
          let embedValue = '';
          while (peek() && peek()!.type !== 'rembed') {
            const embedToken = consume()!;
            embedValue +=
              embedToken.type === 'word' || embedToken.type === 'punct' || embedToken.type === 'whitespace'
                ? embedToken.value
                : '';
          }
          if (peek()?.type === 'rembed') {
            consume(); // consume '>'
          }
          nodes.push({ type: 'embedding', value: embedValue.trim() });
          break;
        }
        case 'word': {
          const wordToken = consume() as Token & { type: 'word' };
          let attention: Attention | undefined;

          // Check for immediate weight after word
          if (peek()?.type === 'weight') {
            attention = (consume() as Token & { type: 'weight' }).value;
          }

          nodes.push({ type: 'word', text: wordToken.value, attention });
          break;
        }
        case 'punct': {
          const punctToken = consume() as Token & { type: 'punct' };
          nodes.push({ type: 'punct', value: punctToken.value });
          break;
        }
        default: {
          consume();
        }
      }
    }

    return nodes;
  }

  return parseGroup();
}

/**
 * Convert an AST back into a prompt string.
 * @param ast ASTNode[]
 * @returns string
 */
export function serialize(ast: ASTNode[]): string {
  let prompt = '';

  for (const node of ast) {
    switch (node.type) {
      case 'punct':
      case 'whitespace': {
        prompt += node.value;
        break;
      }
      case 'word': {
        prompt += node.text;
        if (node.attention) {
          prompt += String(node.attention);
        }
        break;
      }
      case 'group': {
        prompt += '(';
        prompt += serialize(node.children);
        prompt += ')';
        if (node.attention) {
          prompt += String(node.attention);
        }
        break;
      }
      case 'embedding': {
        prompt += `<${node.value}>`;
        break;
      }
    }
  }

  return prompt;
}
