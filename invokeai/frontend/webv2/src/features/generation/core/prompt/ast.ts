export type PromptAttention = string | number;

export interface PromptRange {
  start: number;
  end: number;
}

export type PromptToken =
  | { type: 'word'; value: string; range: PromptRange }
  | { type: 'whitespace'; value: string; range: PromptRange }
  | { type: 'punct'; value: string; range: PromptRange }
  | { type: 'lparen'; range: PromptRange }
  | { type: 'rparen'; range: PromptRange }
  | { type: 'weight'; value: PromptAttention; range: PromptRange }
  | { type: 'lembed'; range: PromptRange }
  | { type: 'rembed'; range: PromptRange }
  | { type: 'escaped_paren'; value: '(' | ')'; range: PromptRange };

export interface PromptFunctionArg {
  nodes: PromptAstNode[];
  quote: string;
  contentRange: PromptRange;
  separator?: string;
}

export type PromptAstNode =
  | { type: 'word'; text: string; attention?: PromptAttention; range: PromptRange; isSelection?: boolean }
  | { type: 'group'; children: PromptAstNode[]; attention?: PromptAttention; range: PromptRange; isSelection?: boolean }
  | { type: 'embedding'; value: string; range: PromptRange; isSelection?: boolean }
  | { type: 'whitespace'; value: string; range: PromptRange; isSelection?: boolean }
  | { type: 'punct'; value: string; range: PromptRange; isSelection?: boolean }
  | { type: 'escaped_paren'; value: '(' | ')'; range: PromptRange; isSelection?: boolean }
  | {
      type: 'prompt_function';
      name: string;
      promptArgs: PromptFunctionArg[];
      functionParams: string;
      range: PromptRange;
      isSelection?: boolean;
    };

const WORD_CHAR = /[A-Za-z0-9_]/;
const WHITESPACE = /\s/;
const NUMERIC_WEIGHT = /^[+-]?\d+(?:\.\d+)?/;
const SYMBOLIC_WEIGHT = /^[+-]+/;
const PUNCTUATION = new Set([
  '.',
  ',',
  '/',
  '!',
  '?',
  ';',
  ':',
  "'",
  '"',
  '\u2018',
  '\u2019',
  '\u201c',
  '\u201d',
  '`',
  '~',
  '@',
  '#',
  '$',
  '%',
  '^',
  '&',
  '*',
  '=',
  '_',
  '|',
  '+',
  '-',
]);

const OPEN_QUOTES = new Set(["'", '"', '\u2018', '\u201c']);
const CLOSE_QUOTE_BY_OPEN: Record<string, string> = {
  "'": "'",
  '"': '"',
  '\u2018': '\u2019',
  '\u201c': '\u201d',
};

const range = (start: number, end: number): PromptRange => ({ start, end });

const isWordChar = (char: string | undefined): boolean => Boolean(char && WORD_CHAR.test(char));

const attentionValue = (raw: string): PromptAttention => {
  const numeric = Number(raw);

  return Number.isNaN(numeric) ? raw : numeric;
};

const readWhitespace = (prompt: string, start: number): PromptToken => {
  let end = start + 1;

  while (end < prompt.length && WHITESPACE.test(prompt[end] ?? '')) {
    end++;
  }

  return { type: 'whitespace', value: prompt.slice(start, end), range: range(start, end) };
};

const readSymbolicAttention = (prompt: string, start: number): { value: string; end: number } | null => {
  const match = prompt.slice(start).match(SYMBOLIC_WEIGHT)?.[0];

  if (!match) {
    return null;
  }

  const end = start + match.length;

  if (isWordChar(prompt[end])) {
    return null;
  }

  return { value: match, end };
};

const readNumericAttention = (prompt: string, start: number): { value: number; end: number } | null => {
  const match = prompt.slice(start).match(NUMERIC_WEIGHT)?.[0];

  if (!match) {
    return null;
  }

  const end = start + match.length;

  if (prompt[end] === '.' || isWordChar(prompt[end])) {
    return null;
  }

  return { value: Number(match), end };
};

const readWord = (prompt: string, start: number): { token: PromptToken; extraToken?: PromptToken; end: number } => {
  let end = start + 1;

  while (end < prompt.length && isWordChar(prompt[end])) {
    end++;
  }

  const token: PromptToken = { type: 'word', value: prompt.slice(start, end), range: range(start, end) };
  const symbolic = readSymbolicAttention(prompt, end);

  if (!symbolic) {
    return { token, end };
  }

  return {
    token,
    extraToken: { type: 'weight', value: symbolic.value, range: range(end, symbolic.end) },
    end: symbolic.end,
  };
};

export const tokenizePrompt = (prompt: string): PromptToken[] => {
  const tokens: PromptToken[] = [];
  let index = 0;

  while (index < prompt.length) {
    const char = prompt[index] ?? '';

    if (WHITESPACE.test(char)) {
      const token = readWhitespace(prompt, index);
      tokens.push(token);
      index = token.range.end;
      continue;
    }

    if (char === '\\' && (prompt[index + 1] === '(' || prompt[index + 1] === ')')) {
      tokens.push({ type: 'escaped_paren', value: prompt[index + 1] as '(' | ')', range: range(index, index + 2) });
      index += 2;
      continue;
    }

    if (char === '(') {
      tokens.push({ type: 'lparen', range: range(index, index + 1) });
      index++;
      continue;
    }

    if (char === ')') {
      tokens.push({ type: 'rparen', range: range(index, index + 1) });

      const numeric = readNumericAttention(prompt, index + 1);
      const symbolic = numeric ? null : readSymbolicAttention(prompt, index + 1);
      const weight = numeric ?? symbolic;

      if (weight) {
        tokens.push({ type: 'weight', value: weight.value, range: range(index + 1, weight.end) });
        index = weight.end;
      } else {
        index++;
      }
      continue;
    }

    if (char === '<') {
      tokens.push({ type: 'lembed', range: range(index, index + 1) });
      index++;
      continue;
    }

    if (char === '>') {
      tokens.push({ type: 'rembed', range: range(index, index + 1) });
      index++;
      continue;
    }

    if (isWordChar(char)) {
      const result = readWord(prompt, index);
      tokens.push(result.token);

      if (result.extraToken) {
        tokens.push(result.extraToken);
      }

      index = result.end;
      continue;
    }

    if (PUNCTUATION.has(char)) {
      tokens.push({ type: 'punct', value: char, range: range(index, index + 1) });
      index++;
      continue;
    }

    tokens.push({ type: 'word', value: char, range: range(index, index + 1) });
    index++;
  }

  return tokens;
};

const tokenValue = (token: PromptToken | undefined): string | undefined => {
  if (!token || !('value' in token)) {
    return undefined;
  }

  return String(token.value);
};

const tokenSource = (token: PromptToken): string => {
  switch (token.type) {
    case 'lparen':
      return '(';
    case 'rparen':
      return ')';
    case 'lembed':
      return '<';
    case 'rembed':
      return '>';
    case 'escaped_paren':
      return `\\${token.value}`;
    case 'weight':
      return String(token.value);
    default:
      return token.value;
  }
};

class PromptParser {
  private index = 0;

  constructor(private readonly tokens: PromptToken[]) {}

  parse(): PromptAstNode[] {
    return this.parseNodes();
  }

  private peek(offset = 0): PromptToken | undefined {
    return this.tokens[this.index + offset];
  }

  private consume(): PromptToken {
    const token = this.tokens[this.index];

    if (!token) {
      throw new Error('Unexpected end of prompt tokens');
    }

    this.index++;
    return token;
  }

  private isPunct(offset: number, value: string): boolean {
    return this.peek(offset)?.type === 'punct' && tokenValue(this.peek(offset)) === value;
  }

  private quotedPromptFunctionAhead(): boolean {
    let offset = 0;

    while (this.peek(offset)?.type === 'whitespace') {
      offset++;
    }

    const token = this.peek(offset);

    return token?.type === 'punct' && OPEN_QUOTES.has(tokenValue(token) ?? '');
  }

  private unquotedPromptFunctionAhead(): boolean {
    let offset = 0;
    let depth = 0;
    let hasComma = false;

    while (this.peek(offset)) {
      const token = this.peek(offset);

      if (token?.type === 'lparen') {
        depth++;
      } else if (token?.type === 'rparen') {
        if (depth === 0) {
          if (!hasComma) {
            return false;
          }

          let tailOffset = offset + 1;

          while (this.peek(tailOffset)?.type === 'whitespace') {
            tailOffset++;
          }

          return (
            this.isPunct(tailOffset, '.') &&
            this.peek(tailOffset + 1)?.type === 'word' &&
            this.peek(tailOffset + 2)?.type === 'lparen'
          );
        }

        depth--;
      } else if (depth === 0 && token?.type === 'punct' && tokenValue(token) === ',') {
        hasComma = true;
      }

      offset++;
    }

    return false;
  }

  private parseMethodTail(savedIndex: number): { name: string; params: string; end: number } | null {
    while (this.peek()?.type === 'whitespace') {
      this.consume();
    }

    if (!this.isPunct(0, '.') || this.peek(1)?.type !== 'word' || this.peek(2)?.type !== 'lparen') {
      this.index = savedIndex;
      return null;
    }

    this.consume();
    const name = tokenValue(this.consume()) ?? '';
    this.consume();

    let depth = 0;
    let params = '';

    while (this.peek()) {
      const token = this.peek();

      if (token?.type === 'rparen' && depth === 0) {
        break;
      }

      const consumed = this.consume();

      if (consumed.type === 'lparen') {
        depth++;
      } else if (consumed.type === 'rparen') {
        depth--;
      }

      params += tokenSource(consumed);
    }

    if (this.peek()?.type !== 'rparen') {
      this.index = savedIndex;
      return null;
    }

    const close = this.consume();

    return { name, params, end: close.range.end };
  }

  private parseQuotedPromptFunction(
    lparen: PromptToken & { type: 'lparen' },
    savedIndex: number
  ): PromptAstNode | null {
    const args: PromptFunctionArg[] = [];
    let expectedOpenQuote: string | null = null;
    let pendingSeparator: string | undefined;

    while (this.peek()) {
      while (this.peek()?.type === 'whitespace') {
        this.consume();
      }

      if (this.peek()?.type === 'rparen') {
        break;
      }

      if (args.length > 0) {
        if (!this.isPunct(0, ',')) {
          this.index = savedIndex;
          return null;
        }

        this.consume();
        pendingSeparator = this.consumeWhitespaceValue();
      }

      const openQuote = this.peek();
      const quote = tokenValue(openQuote);

      if (openQuote?.type !== 'punct' || !quote || !OPEN_QUOTES.has(quote)) {
        this.index = savedIndex;
        return null;
      }

      if (expectedOpenQuote === null) {
        expectedOpenQuote = quote;
      } else if (quote !== expectedOpenQuote) {
        this.index = savedIndex;
        return null;
      }

      this.consume();

      const closeQuote = CLOSE_QUOTE_BY_OPEN[quote] ?? quote;
      const contentStart = openQuote.range.end;
      const argTokens: PromptToken[] = [];
      let contentEnd = contentStart;

      while (this.peek() && !(this.peek()?.type === 'punct' && tokenValue(this.peek()) === closeQuote)) {
        const token = this.consume();
        argTokens.push(token);
        contentEnd = token.range.end;
      }

      if (!(this.peek()?.type === 'punct' && tokenValue(this.peek()) === closeQuote)) {
        this.index = savedIndex;
        return null;
      }

      this.consume();
      args.push({
        contentRange: range(contentStart, contentEnd),
        nodes: parsePromptTokens(argTokens),
        quote,
        separator: pendingSeparator,
      });
      pendingSeparator = undefined;
    }

    if (args.length === 0 || this.peek()?.type !== 'rparen') {
      this.index = savedIndex;
      return null;
    }

    this.consume();

    const tail = this.parseMethodTail(savedIndex);

    if (!tail) {
      return null;
    }

    return {
      type: 'prompt_function',
      name: tail.name,
      promptArgs: args,
      functionParams: tail.params,
      range: range(lparen.range.start, tail.end),
    };
  }

  private parseUnquotedPromptFunction(
    lparen: PromptToken & { type: 'lparen' },
    savedIndex: number
  ): PromptAstNode | null {
    const args: PromptFunctionArg[] = [];
    let pendingSeparator: string | undefined;

    while (this.peek()) {
      if (this.peek()?.type === 'rparen') {
        break;
      }

      if (args.length > 0) {
        if (!this.isPunct(0, ',')) {
          this.index = savedIndex;
          return null;
        }

        this.consume();
        pendingSeparator = this.consumeWhitespaceValue();
      }

      const argTokens = this.consumeUnquotedArgTokens();

      if (argTokens.length === 0) {
        this.index = savedIndex;
        return null;
      }

      const trimmedTokens = trimWhitespaceTokens(argTokens);
      const first = trimmedTokens[0];
      const last = trimmedTokens.at(-1);

      if (!first || !last) {
        this.index = savedIndex;
        return null;
      }

      args.push({
        contentRange: range(first.range.start, last.range.end),
        nodes: parsePromptTokens(trimmedTokens),
        quote: '',
        separator: pendingSeparator,
      });
      pendingSeparator = undefined;
    }

    if (args.length < 2 || this.peek()?.type !== 'rparen') {
      this.index = savedIndex;
      return null;
    }

    this.consume();

    const tail = this.parseMethodTail(savedIndex);

    if (!tail) {
      return null;
    }

    return {
      type: 'prompt_function',
      name: tail.name,
      promptArgs: args,
      functionParams: tail.params,
      range: range(lparen.range.start, tail.end),
    };
  }

  private consumeWhitespaceValue(): string {
    let value = '';

    while (this.peek()?.type === 'whitespace') {
      value += tokenValue(this.consume()) ?? '';
    }

    return value;
  }

  private consumeUnquotedArgTokens(): PromptToken[] {
    const tokens: PromptToken[] = [];
    let depth = 0;

    while (this.peek()) {
      const token = this.peek();

      if (token?.type === 'lparen') {
        depth++;
      } else if (token?.type === 'rparen') {
        if (depth === 0) {
          break;
        }

        depth--;
      } else if (depth === 0 && token?.type === 'punct' && tokenValue(token) === ',') {
        break;
      }

      tokens.push(this.consume());
    }

    return tokens;
  }

  private parseNodes(): PromptAstNode[] {
    const nodes: PromptAstNode[] = [];

    while (this.peek() && this.peek()?.type !== 'rparen') {
      const token = this.consume();

      switch (token.type) {
        case 'whitespace':
          nodes.push({ type: 'whitespace', value: token.value, range: token.range });
          break;
        case 'lparen': {
          const lparen = token;
          const savedIndex = this.index;
          const promptFunction = this.parsePromptFunctionIfPresent(lparen, savedIndex);

          if (promptFunction) {
            nodes.push(promptFunction);
            break;
          }

          const children = this.parseNodes();
          let nodeEnd = lparen.range.end;
          let attention: PromptAttention | undefined;

          if (this.peek()?.type === 'rparen') {
            const rparen = this.consume();
            nodeEnd = rparen.range.end;

            if (this.peek()?.type === 'weight') {
              const weight = this.consume() as PromptToken & { type: 'weight' };
              attention = weight.value;
              nodeEnd = weight.range.end;
            }
          } else if (children.length > 0) {
            nodeEnd = children.at(-1)?.range.end ?? nodeEnd;
          }

          nodes.push({ type: 'group', attention, children, range: range(lparen.range.start, nodeEnd) });
          break;
        }
        case 'lembed': {
          let value = '';
          let nodeEnd = token.range.end;

          while (this.peek() && this.peek()?.type !== 'rembed') {
            const embedToken = this.consume();
            value += tokenSource(embedToken);
            nodeEnd = embedToken.range.end;
          }

          if (this.peek()?.type === 'rembed') {
            nodeEnd = this.consume().range.end;
          }

          nodes.push({ type: 'embedding', value: value.trim(), range: range(token.range.start, nodeEnd) });
          break;
        }
        case 'word': {
          let nodeEnd = token.range.end;
          let attention: PromptAttention | undefined;

          if (this.peek()?.type === 'weight') {
            const weight = this.consume() as PromptToken & { type: 'weight' };
            attention = attentionValue(String(weight.value));
            nodeEnd = weight.range.end;
          }

          nodes.push({ type: 'word', text: token.value, attention, range: range(token.range.start, nodeEnd) });
          break;
        }
        case 'punct':
          nodes.push({ type: 'punct', value: token.value, range: token.range });
          break;
        case 'escaped_paren':
          nodes.push({ type: 'escaped_paren', value: token.value, range: token.range });
          break;
        default:
          break;
      }
    }

    return nodes;
  }

  private parsePromptFunctionIfPresent(
    lparen: PromptToken & { type: 'lparen' },
    savedIndex: number
  ): PromptAstNode | null {
    if (this.quotedPromptFunctionAhead()) {
      const quoted = this.parseQuotedPromptFunction(lparen, savedIndex);

      if (quoted) {
        return quoted;
      }
    }

    if (this.unquotedPromptFunctionAhead()) {
      const unquoted = this.parseUnquotedPromptFunction(lparen, savedIndex);

      if (unquoted) {
        return unquoted;
      }
    }

    this.index = savedIndex;
    return null;
  }
}

const trimWhitespaceTokens = (tokens: PromptToken[]): PromptToken[] => {
  let start = 0;
  let end = tokens.length;

  while (start < end && tokens[start]?.type === 'whitespace') {
    start++;
  }

  while (end > start && tokens[end - 1]?.type === 'whitespace') {
    end--;
  }

  return tokens.slice(start, end);
};

export const parsePromptTokens = (tokens: PromptToken[]): PromptAstNode[] => new PromptParser(tokens).parse();

export const parsePrompt = (prompt: string): PromptAstNode[] => parsePromptTokens(tokenizePrompt(prompt));

interface SerializeVisitor {
  onNode?: (node: PromptAstNode, start: number, end: number) => void;
}

const appendAttention = (attention: PromptAttention | undefined): string =>
  attention === undefined ? '' : String(attention);

const serializeInto = (nodes: PromptAstNode[], output: { value: string }, visitor?: SerializeVisitor): void => {
  for (const node of nodes) {
    const start = output.value.length;

    switch (node.type) {
      case 'punct':
      case 'whitespace':
        output.value += node.value;
        break;
      case 'escaped_paren':
        output.value += `\\${node.value}`;
        break;
      case 'word':
        output.value += `${node.text}${appendAttention(node.attention)}`;
        break;
      case 'group':
        output.value += '(';
        serializeInto(node.children, output, visitor);
        output.value += `)${appendAttention(node.attention)}`;
        break;
      case 'embedding':
        output.value += `<${node.value}>`;
        break;
      case 'prompt_function':
        output.value += '(';
        node.promptArgs.forEach((arg, index) => {
          if (index > 0) {
            output.value += `,${arg.separator ?? ' '}`;
          }

          output.value += arg.quote;
          serializeInto(arg.nodes, output, visitor);
          output.value += CLOSE_QUOTE_BY_OPEN[arg.quote] ?? arg.quote;
        });
        output.value += `).${node.name}(${node.functionParams})`;
        break;
    }

    visitor?.onNode?.(node, start, output.value.length);
  }
};

export const serializePrompt = (nodes: PromptAstNode[]): string => {
  const output = { value: '' };

  serializeInto(nodes, output);

  return output.value;
};

export const serializePromptWithSelection = (
  nodes: PromptAstNode[]
): { prompt: string; selectionStart: number; selectionEnd: number } => {
  const output = { value: '' };
  let selectionStart = Infinity;
  let selectionEnd = -1;

  serializeInto(nodes, output, {
    onNode: (node, start, end) => {
      if (!node.isSelection) {
        return;
      }

      selectionStart = Math.min(selectionStart, start);
      selectionEnd = Math.max(selectionEnd, end);
    },
  });

  if (selectionStart === Infinity) {
    return { prompt: output.value, selectionStart: 0, selectionEnd: output.value.length };
  }

  return { prompt: output.value, selectionStart, selectionEnd };
};
