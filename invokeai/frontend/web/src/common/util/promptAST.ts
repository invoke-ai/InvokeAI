/**
 * Expected as either '+', '-', '++', '--', etc. or a numeric string like '1.2', '0.8', etc.
 */
export type Attention = string | number;

type Word = string;

type Punct = string;

type Whitespace = string;

type Embedding = string;

type Token =
  | { type: 'word'; value: Word; start: number; end: number }
  | { type: 'whitespace'; value: Whitespace; start: number; end: number }
  | { type: 'punct'; value: Punct; start: number; end: number }
  | { type: 'lparen'; start: number; end: number }
  | { type: 'rparen'; start: number; end: number }
  | { type: 'weight'; value: Attention; start: number; end: number }
  | { type: 'lembed'; start: number; end: number }
  | { type: 'rembed'; start: number; end: number }
  | { type: 'escaped_paren'; value: '(' | ')'; start: number; end: number };

export type ASTNode =
  | { type: 'word'; text: Word; attention?: Attention; range: { start: number; end: number }; isSelection?: boolean }
  | {
      type: 'group';
      children: ASTNode[];
      attention?: Attention;
      range: { start: number; end: number };
      isSelection?: boolean;
    }
  | { type: 'embedding'; value: Embedding; range: { start: number; end: number }; isSelection?: boolean }
  | { type: 'whitespace'; value: Whitespace; range: { start: number; end: number }; isSelection?: boolean }
  | { type: 'punct'; value: Punct; range: { start: number; end: number }; isSelection?: boolean }
  | { type: 'escaped_paren'; value: '(' | ')'; range: { start: number; end: number }; isSelection?: boolean };

const WEIGHT_PATTERN = /^[+-]?(\d+(\.\d+)?|[+-]+)/;
const WHITESPACE_PATTERN = /^\s+/;
const PUNCTUATION_PATTERN = /^[.,]/;
const OTHER_PATTERN = /\s/;

/**
 * Convert a prompt string into an AST.
 * @param prompt string
 * @returns ASTNode[]
 */
export function tokenize(prompt: string): Token[] {
  if (!prompt) {
    return [];
  }

  const len = prompt.length;
  let tokens: Token[] = [];
  let i = 0;

  while (i < len) {
    const char = prompt[i];
    if (!char) {
      break;
    }

    const result =
      tokenizeWhitespace(char, i) ||
      tokenizeEscapedParen(prompt, i) ||
      tokenizeLeftParen(char, i) ||
      tokenizeRightParen(prompt, i) ||
      tokenizeEmbedding(char, i) ||
      tokenizeWord(prompt, i) ||
      tokenizePunctuation(char, i) ||
      tokenizeOther(char, i);

    if (result) {
      if (result.token) {
        tokens.push(result.token);
      }
      if (result.extraToken) {
        tokens.push(result.extraToken);
      }
      i = result.nextIndex;
    } else {
      i++;
    }
  }

  return tokens;
}

type TokenizeResult = {
  token?: Token;
  extraToken?: Token;
  nextIndex: number;
} | null;

function tokenizeWhitespace(char: string, i: number): TokenizeResult {
  if (WHITESPACE_PATTERN.test(char)) {
    return {
      token: { type: 'whitespace', value: char, start: i, end: i + 1 },
      nextIndex: i + 1,
    };
  }
  return null;
}

function tokenizeEscapedParen(prompt: string, i: number): TokenizeResult {
  const char = prompt[i];
  if (char === '\\' && i + 1 < prompt.length) {
    const nextChar = prompt[i + 1];
    if (nextChar === '(' || nextChar === ')') {
      return {
        token: { type: 'escaped_paren', value: nextChar, start: i, end: i + 2 },
        nextIndex: i + 2,
      };
    }
  }
  return null;
}

function tokenizeLeftParen(char: string, i: number): TokenizeResult {
  if (char === '(') {
    return {
      token: { type: 'lparen', start: i, end: i + 1 },
      nextIndex: i + 1,
    };
  }
  return null;
}

function tokenizeRightParen(prompt: string, i: number): TokenizeResult {
  const char = prompt[i];
  if (char === ')') {
    // Look ahead for weight like ')1.1' or ')-0.9' or ')+' or ')-'
    const weightMatch = prompt.slice(i + 1).match(WEIGHT_PATTERN);
    if (weightMatch && weightMatch[0]) {
      let weight: Attention = weightMatch[0];
      if (!isNaN(Number(weight))) {
        weight = Number(weight);
      }
      const weightEnd = i + 1 + weightMatch[0].length;
      return {
        token: { type: 'rparen', start: i, end: i + 1 },
        extraToken: { type: 'weight', value: weight, start: i + 1, end: weightEnd },
        nextIndex: weightEnd,
      };
    }
    return {
      token: { type: 'rparen', start: i, end: i + 1 },
      nextIndex: i + 1,
    };
  }
  return null;
}

function tokenizePunctuation(char: string, i: number): TokenizeResult {
  if (PUNCTUATION_PATTERN.test(char)) {
    return {
      token: { type: 'punct', value: char, start: i, end: i + 1 },
      nextIndex: i + 1,
    };
  }
  return null;
}

function tokenizeWord(prompt: string, i: number): TokenizeResult {
  const char = prompt[i];
  if (!char) {
    return null;
  }

  if (/[a-zA-Z0-9_]/.test(char)) {
    let j = i;
    while (j < prompt.length && /[a-zA-Z0-9_]/.test(prompt[j]!)) {
      j++;
    }
    const word = prompt.slice(i, j);

    // Check for weight immediately after word (e.g., "Lorem+", "consectetur-")
    const weightMatch = prompt.slice(j).match(/^[+-]?(\d+(\.\d+)?|[+-]+)/);
    if (weightMatch && weightMatch[0]) {
      const weightEnd = j + weightMatch[0].length;
      return {
        token: { type: 'word', value: word, start: i, end: j },
        extraToken: { type: 'weight', value: weightMatch[0], start: j, end: weightEnd },
        nextIndex: weightEnd,
      };
    }

    return {
      token: { type: 'word', value: word, start: i, end: j },
      nextIndex: j,
    };
  }
  return null;
}

function tokenizeEmbedding(char: string, i: number): TokenizeResult {
  if (char === '<') {
    return {
      token: { type: 'lembed', start: i, end: i + 1 },
      nextIndex: i + 1,
    };
  }
  if (char === '>') {
    return {
      token: { type: 'rembed', start: i, end: i + 1 },
      nextIndex: i + 1,
    };
  }
  return null;
}

function tokenizeOther(char: string, i: number): TokenizeResult {
  // Any other single character punctuation
  if (OTHER_PATTERN.test(char)) {
    return {
      token: { type: 'punct', value: char, start: i, end: i + 1 },
      nextIndex: i + 1,
    };
  }
  return null;
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

      switch (token.type) {
        case 'whitespace': {
          const wsToken = consume() as Token & { type: 'whitespace' };
          nodes.push({ type: 'whitespace', value: wsToken.value, range: { start: wsToken.start, end: wsToken.end } });
          break;
        }
        case 'lparen': {
          const lparen = consume() as Token & { type: 'lparen' };
          const groupChildren = parseGroup();

          let attention: Attention | undefined;
          let end = lparen.end; // Default end if no rparen

          if (peek()?.type === 'rparen') {
            const rparen = consume() as Token & { type: 'rparen' };
            end = rparen.end;
            if (peek()?.type === 'weight') {
              const weightToken = consume() as Token & { type: 'weight' };
              attention = weightToken.value;
              end = weightToken.end;
            }
          }

          // If we hit EOF without rparen, the group extends to the end of the last child
          if (end === lparen.end && groupChildren.length > 0) {
            end = groupChildren[groupChildren.length - 1]!.range.end;
          }

          nodes.push({ type: 'group', children: groupChildren, attention, range: { start: lparen.start, end } });
          break;
        }
        case 'lembed': {
          const lembed = consume() as Token & { type: 'lembed' };
          let embedValue = '';
          let end = lembed.end;
          while (peek() && peek()!.type !== 'rembed') {
            const embedToken = consume()!;
            embedValue +=
              embedToken.type === 'word' || embedToken.type === 'punct' || embedToken.type === 'whitespace'
                ? embedToken.value
                : '';
            end = embedToken.end;
          }
          if (peek()?.type === 'rembed') {
            const rembed = consume() as Token & { type: 'rembed' };
            end = rembed.end;
          }
          nodes.push({ type: 'embedding', value: embedValue.trim(), range: { start: lembed.start, end } });
          break;
        }
        case 'word': {
          const wordToken = consume() as Token & { type: 'word' };
          let attention: Attention | undefined;
          let end = wordToken.end;

          // Check for immediate weight after word
          if (peek()?.type === 'weight') {
            const weightToken = consume() as Token & { type: 'weight' };
            attention = weightToken.value;
            end = weightToken.end;
          }

          nodes.push({ type: 'word', text: wordToken.value, attention, range: { start: wordToken.start, end } });
          break;
        }
        case 'punct': {
          const punctToken = consume() as Token & { type: 'punct' };
          nodes.push({
            type: 'punct',
            value: punctToken.value,
            range: { start: punctToken.start, end: punctToken.end },
          });
          break;
        }
        case 'escaped_paren': {
          const escapedToken = consume() as Token & { type: 'escaped_paren' };
          nodes.push({
            type: 'escaped_paren',
            value: escapedToken.value,
            range: { start: escapedToken.start, end: escapedToken.end },
          });
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
      case 'escaped_paren': {
        prompt += `\\${node.value}`;
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
