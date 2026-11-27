/**
 * Expected as either '+', '-', '++', '--', etc. or a numeric string like '1.2', '0.8', etc.
 */
export type Attention = string | number;

type Token =
  | { type: 'word'; value: string; start: number; end: number }
  | { type: 'whitespace'; value: string; start: number; end: number }
  | { type: 'punct'; value: string; start: number; end: number }
  | { type: 'lparen'; start: number; end: number }
  | { type: 'rparen'; start: number; end: number }
  | { type: 'weight'; value: Attention; start: number; end: number }
  | { type: 'lembed'; start: number; end: number }
  | { type: 'rembed'; start: number; end: number }
  | { type: 'escaped_paren'; value: '(' | ')'; start: number; end: number };

/**
 * A single argument in a prompt function like .and(), .or(), or .blend().
 * Contains the parsed AST nodes of the argument content and metadata about quoting/range.
 */
export type PromptFunctionArg = {
  nodes: ASTNode[];
  quote: string;
  /** Range of the content between the quotes (exclusive of quotes themselves) in original prompt coordinates. */
  contentRange: { start: number; end: number };
  /** Raw separator whitespace after the comma before this arg (args[1+] only). */
  separator?: string;
};

export type ASTNode =
  | { type: 'word'; text: string; attention?: Attention; range: { start: number; end: number }; isSelection?: boolean }
  | {
      type: 'group';
      children: ASTNode[];
      attention?: Attention;
      range: { start: number; end: number };
      isSelection?: boolean;
    }
  | { type: 'embedding'; value: string; range: { start: number; end: number }; isSelection?: boolean }
  | { type: 'whitespace'; value: string; range: { start: number; end: number }; isSelection?: boolean }
  | { type: 'punct'; value: string; range: { start: number; end: number }; isSelection?: boolean }
  | { type: 'escaped_paren'; value: '(' | ')'; range: { start: number; end: number }; isSelection?: boolean }
  | {
      type: 'prompt_function';
      name: string;
      promptArgs: PromptFunctionArg[];
      functionParams: string;
      range: { start: number; end: number };
      isSelection?: boolean;
    };

const WEIGHT_PATTERN = /^[+-]?(\d+(\.\d+)?|[+-]+)/;
const WHITESPACE_PATTERN = /^\s+/;
const WORD_CHAR_PATTERN = /[a-zA-Z0-9_]/;
// prettier-ignore
const PUNCTUATION_PATTERN = /^[.,/!?;:'"""''\u2018\u2019\u201c\u201d`~@#$%^&*=_|]/;

/** All characters that can serve as an opening quote in a prompt function argument. */
const OPEN_QUOTE_CHARS = new Set(["'", '"', '\u2018', '\u201c']);

/** Map from opening curly quote to the matching closing curly quote. Straight quotes match themselves. */
const CLOSE_QUOTE_MAP: Record<string, string> = {
  "'": "'",
  '"': '"',
  '\u2018': '\u2019', // ' → '
  '\u201c': '\u201d', // " → "
};

// #region Token Helpers

/** Get the string value of a token, if it has one. */
function tokenValue(t: Token | undefined): string | undefined {
  if (!t) {
    return undefined;
  }
  if ('value' in t) {
    return String(t.value);
  }
  return undefined;
}

/** Check if a token is a punct token with a specific value. */
function isPunctValue(t: Token | undefined, value: string): boolean {
  return t?.type === 'punct' && tokenValue(t) === value;
}

// #region Tokenizer

/**
 * Convert a prompt string into a token stream.
 * @param prompt string
 * @returns Token[]
 */
export function tokenize(prompt: string): Token[] {
  if (!prompt) {
    return [];
  }

  const len = prompt.length;
  const tokens: Token[] = [];
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
      tokenizeFallback(char, i);

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

  if (WORD_CHAR_PATTERN.test(char)) {
    let j = i;
    while (j < prompt.length && WORD_CHAR_PATTERN.test(prompt[j]!)) {
      j++;
    }
    const word = prompt.slice(i, j);

    // Check for weight immediately after word (e.g., "Lorem+", "consectetur-")
    const weightMatch = prompt.slice(j).match(WEIGHT_PATTERN);
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

/**
 * Fallback tokenizer for characters not matched by any other tokenizer.
 * Emits them as word tokens so they are preserved in the AST rather than silently dropped.
 * This handles non-Latin Unicode text (CJK, emoji, etc.) and any other unrecognized characters.
 */
function tokenizeFallback(char: string, i: number): TokenizeResult {
  return {
    token: { type: 'word', value: char, start: i, end: i + 1 },
    nextIndex: i + 1,
  };
}

// #region Parser

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

  function peekAt(offset: number): Token | undefined {
    return tokens[pos + offset];
  }

  function consume(): Token | undefined {
    return tokens[pos++];
  }

  /**
   * Quick lookahead check: does the current lparen (already consumed) start a quoted prompt function?
   * A quoted prompt function looks like ('...', '...').method(...)
   * We check if the first non-whitespace token after lparen is a quote character.
   */
  function isQuotedPromptFunctionAhead(): boolean {
    let p = 0;
    while (peekAt(p)?.type === 'whitespace') {
      p++;
    }
    const t = peekAt(p);
    return t?.type === 'punct' && OPEN_QUOTE_CHARS.has(tokenValue(t)!);
  }

  /**
   * Lookahead check: does the current lparen (already consumed) start an unquoted prompt function?
   * An unquoted prompt function looks like (arg1, arg2).method(...) where args are not quoted.
   * We scan forward looking for a comma at the same nesting depth, then rparen followed by .word(
   */
  function isUnquotedPromptFunctionAhead(): boolean {
    let p = 0;
    let depth = 0;
    let hasComma = false;

    // Scan forward through tokens to find the matching rparen
    while (peekAt(p)) {
      const t = peekAt(p)!;

      if (t.type === 'lparen') {
        depth++;
      } else if (t.type === 'rparen') {
        if (depth === 0) {
          // Found matching rparen — now check for .methodName( pattern
          // (possibly with whitespace between ) and .)
          if (!hasComma) {
            return false; // No comma means it's just a regular group
          }
          let next = p + 1;
          while (peekAt(next)?.type === 'whitespace') {
            next++;
          }
          return (
            isPunctValue(peekAt(next), '.') && peekAt(next + 1)?.type === 'word' && peekAt(next + 2)?.type === 'lparen'
          );
        }
        depth--;
      } else if (isPunctValue(t, ',') && depth === 0) {
        hasComma = true;
      }

      p++;
    }
    return false;
  }

  /**
   * Parse the `.methodName(params)` suffix that follows the closing rparen of a prompt function.
   * Assumes whitespace has already been skipped. Returns null and restores pos if the pattern
   * doesn't match.
   */
  function tryParseMethodTail(savedPos: number): { name: string; functionParams: string; endPos: number } | null {
    // Skip whitespace between ) and .methodName (allows newlines)
    while (peek()?.type === 'whitespace') {
      consume();
    }

    // Expect .methodName(params)
    if (!isPunctValue(peek(), '.')) {
      pos = savedPos;
      return null;
    }
    consume(); // consume dot

    if (peek()?.type !== 'word') {
      pos = savedPos;
      return null;
    }
    const methodName = tokenValue(consume())!;

    // Expect opening paren for method call
    if (peek()?.type !== 'lparen') {
      pos = savedPos;
      return null;
    }
    consume(); // consume method open paren

    // Collect method params until closing rparen
    let functionParams = '';
    while (pos < tokens.length) {
      const t = peek()!;
      if (t.type === 'rparen') {
        break;
      }
      const tok = consume()!;
      const v = tokenValue(tok);
      if (v !== undefined) {
        functionParams += v;
      }
    }

    // Expect closing rparen for method call
    if (peek()?.type !== 'rparen') {
      pos = savedPos;
      return null;
    }
    const methodCloseParen = consume()!; // consume method close paren

    return { name: methodName, functionParams, endPos: methodCloseParen.end };
  }

  /**
   * Try to parse a prompt function starting after the opening lparen.
   * Returns the PromptFunctionNode if successful, or null if the pattern doesn't match
   * (in which case `pos` is restored to `savedPos`).
   */
  function tryParsePromptFunction(lparenToken: Token & { type: 'lparen' }, savedPos: number): ASTNode | null {
    const args: PromptFunctionArg[] = [];
    let openQuoteChar: string | null = null;
    let closeQuoteChar: string | null = null;
    let pendingSeparator: string | undefined;

    while (pos < tokens.length) {
      // Skip whitespace before arg or closing paren
      while (peek()?.type === 'whitespace') {
        consume();
      }

      // Check for rparen (end of prompt function args)
      if (peek()?.type === 'rparen') {
        break;
      }

      // Expect comma separator between args
      if (args.length > 0) {
        if (isPunctValue(peek(), ',')) {
          consume();
          let sep = '';
          while (peek()?.type === 'whitespace') {
            const sepToken = consume()!;
            const sepValue = tokenValue(sepToken);
            if (sepValue !== undefined) {
              sep += sepValue;
            }
          }
          pendingSeparator = sep;
        } else {
          pos = savedPos;
          return null;
        }
      }

      // Expect opening quote
      const openQuoteTok = peek();
      if (!openQuoteTok || openQuoteTok.type !== 'punct') {
        pos = savedPos;
        return null;
      }
      const thisOpenQuote = tokenValue(openQuoteTok)!;
      if (!OPEN_QUOTE_CHARS.has(thisOpenQuote)) {
        pos = savedPos;
        return null;
      }

      const thisCloseQuote = CLOSE_QUOTE_MAP[thisOpenQuote]!;
      if (openQuoteChar === null) {
        openQuoteChar = thisOpenQuote;
        closeQuoteChar = thisCloseQuote;
      } else if (thisOpenQuote !== openQuoteChar) {
        // Mismatched quote style between args
        pos = savedPos;
        return null;
      }

      consume(); // consume opening quote
      const contentStart = openQuoteTok.end;

      // Collect tokens until closing quote
      const argTokens: Token[] = [];
      let contentEnd = contentStart;
      while (pos < tokens.length) {
        const t = peek();
        if (isPunctValue(t, closeQuoteChar!)) {
          contentEnd = t!.start;
          break;
        }
        const consumed = consume()!;
        argTokens.push(consumed);
        contentEnd = consumed.end;
      }

      // Expect closing quote
      if (!isPunctValue(peek(), closeQuoteChar!)) {
        pos = savedPos;
        return null;
      }
      consume(); // consume closing quote

      // Parse sub-tokens as AST
      const argNodes = parseTokens(argTokens);

      args.push({
        nodes: argNodes,
        quote: openQuoteChar,
        contentRange: { start: contentStart, end: contentEnd },
        separator: pendingSeparator,
      });
      pendingSeparator = undefined;
    }

    if (args.length === 0) {
      pos = savedPos;
      return null;
    }

    // Expect rparen
    if (peek()?.type !== 'rparen') {
      pos = savedPos;
      return null;
    }
    consume(); // consume rparen

    // Parse .methodName(params) suffix
    const methodTail = tryParseMethodTail(savedPos);
    if (!methodTail) {
      return null; // pos already restored by tryParseMethodTail
    }

    return {
      type: 'prompt_function',
      name: methodTail.name,
      promptArgs: args,
      functionParams: methodTail.functionParams,
      range: { start: lparenToken.start, end: methodTail.endPos },
    };
  }

  /**
   * Try to parse an unquoted prompt function starting after the opening lparen.
   * Unquoted prompt functions look like (arg1 words, arg2 words).method(params)
   * where arguments are separated by commas without quotes.
   * Returns the PromptFunctionNode if successful, or null if the pattern doesn't match
   * (in which case `pos` is restored to `savedPos`).
   */
  function tryParseUnquotedPromptFunction(lparenToken: Token & { type: 'lparen' }, savedPos: number): ASTNode | null {
    const args: PromptFunctionArg[] = [];
    let pendingSeparator: string | undefined;

    while (pos < tokens.length) {
      // Check for rparen (end of prompt function args)
      if (peek()?.type === 'rparen') {
        break;
      }

      // Expect comma separator between args (consume the comma)
      if (args.length > 0) {
        if (isPunctValue(peek(), ',')) {
          consume(); // consume comma
          let sep = '';
          while (peek()?.type === 'whitespace') {
            const sepToken = consume()!;
            const sepValue = tokenValue(sepToken);
            if (sepValue !== undefined) {
              sep += sepValue;
            }
          }
          pendingSeparator = sep;
        } else {
          pos = savedPos;
          return null;
        }
      }

      // Collect tokens until comma or rparen (at nesting depth 0)
      const argTokens: Token[] = [];
      let contentStart: number | null = null;
      let contentEnd: number | null = null;
      let depth = 0;

      while (pos < tokens.length) {
        const t = peek()!;

        if (t.type === 'lparen') {
          depth++;
        } else if (t.type === 'rparen') {
          if (depth === 0) {
            break; // End of all args
          }
          depth--;
        } else if (isPunctValue(t, ',') && depth === 0) {
          break; // End of this arg
        }

        if (contentStart === null) {
          contentStart = t.start;
        }
        const consumed = consume()!;
        argTokens.push(consumed);
        contentEnd = consumed.end;
      }

      if (argTokens.length === 0) {
        pos = savedPos;
        return null;
      }

      // Trim leading/trailing whitespace tokens from the arg content
      let firstNonWs = 0;
      while (firstNonWs < argTokens.length && argTokens[firstNonWs]!.type === 'whitespace') {
        firstNonWs++;
      }
      let lastNonWs = argTokens.length - 1;
      while (lastNonWs >= 0 && argTokens[lastNonWs]!.type === 'whitespace') {
        lastNonWs--;
      }

      const trimmedArgTokens = argTokens.slice(firstNonWs, lastNonWs + 1);
      const trimmedStart = trimmedArgTokens.length > 0 ? trimmedArgTokens[0]!.start : contentStart!;
      const trimmedEnd = trimmedArgTokens.length > 0 ? trimmedArgTokens[trimmedArgTokens.length - 1]!.end : contentEnd!;

      // Parse sub-tokens as AST
      const argNodes = parseTokens(trimmedArgTokens);

      args.push({
        nodes: argNodes,
        quote: '', // Unquoted
        contentRange: { start: trimmedStart, end: trimmedEnd },
        separator: pendingSeparator,
      });
      pendingSeparator = undefined;
    }

    if (args.length < 2) {
      // An unquoted prompt function must have at least 2 args (otherwise it's a regular group)
      pos = savedPos;
      return null;
    }

    // Expect rparen
    if (peek()?.type !== 'rparen') {
      pos = savedPos;
      return null;
    }
    consume(); // consume rparen

    // Parse .methodName(params) suffix
    const methodTail = tryParseMethodTail(savedPos);
    if (!methodTail) {
      return null; // pos already restored by tryParseMethodTail
    }

    return {
      type: 'prompt_function',
      name: methodTail.name,
      promptArgs: args,
      functionParams: methodTail.functionParams,
      range: { start: lparenToken.start, end: methodTail.endPos },
    };
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

          // Try to parse as a quoted prompt function first
          if (isQuotedPromptFunctionAhead()) {
            const savedPos = pos;
            const pfResult = tryParsePromptFunction(lparen, savedPos);
            if (pfResult) {
              nodes.push(pfResult);
              break;
            }
            // pos was restored by tryParsePromptFunction on failure
          }

          // Try to parse as an unquoted prompt function
          if (isUnquotedPromptFunctionAhead()) {
            const savedPos = pos;
            const pfResult = tryParseUnquotedPromptFunction(lparen, savedPos);
            if (pfResult) {
              nodes.push(pfResult);
              break;
            }
            // pos was restored by tryParseUnquotedPromptFunction on failure
          }

          // Regular group parsing
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
            const v = tokenValue(embedToken);
            if (v !== undefined) {
              embedValue += v;
            }
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

// #region Serialization

/**
 * Visitor callbacks for AST serialization. All callbacks are optional.
 * Called during traversal to allow tracking node positions in the output string.
 */
type SerializeVisitor = {
  /** Called after a node has been fully serialized, with its start and end positions in the output. */
  onNode?: (node: ASTNode, start: number, end: number) => void;
};

/** Mutable buffer used by serializeCore so all recursive calls share the same position tracking. */
type SerializeBuffer = { prompt: string };

/**
 * Shared serialization core. Converts an AST back into a prompt string,
 * optionally calling visitor hooks for position tracking.
 *
 * Uses a shared mutable buffer so that node positions reported via
 * `visitor.onNode` are always absolute offsets in the final output string,
 * even for nodes nested inside groups or prompt function args.
 */
function serializeCore(ast: ASTNode[], visitor: SerializeVisitor | undefined, buf: SerializeBuffer): void {
  for (const node of ast) {
    const nodeStart = buf.prompt.length;

    switch (node.type) {
      case 'punct':
      case 'whitespace': {
        buf.prompt += node.value;
        break;
      }
      case 'escaped_paren': {
        buf.prompt += `\\${node.value}`;
        break;
      }
      case 'word': {
        buf.prompt += node.text;
        if (node.attention) {
          buf.prompt += String(node.attention);
        }
        break;
      }
      case 'group': {
        buf.prompt += '(';
        serializeCore(node.children, visitor, buf);
        buf.prompt += ')';
        if (node.attention) {
          buf.prompt += String(node.attention);
        }
        break;
      }
      case 'embedding': {
        buf.prompt += `<${node.value}>`;
        break;
      }
      case 'prompt_function': {
        buf.prompt += '(';
        for (let i = 0; i < node.promptArgs.length; i++) {
          if (i > 0) {
            const sep = node.promptArgs[i]!.separator ?? ' ';
            buf.prompt += `,${sep}`;
          }
          const arg = node.promptArgs[i]!;
          buf.prompt += arg.quote;
          serializeCore(arg.nodes, visitor, buf);
          buf.prompt += CLOSE_QUOTE_MAP[arg.quote] ?? arg.quote;
        }
        buf.prompt += ').';
        buf.prompt += node.name;
        buf.prompt += '(';
        buf.prompt += node.functionParams;
        buf.prompt += ')';
        break;
      }
    }

    visitor?.onNode?.(node, nodeStart, buf.prompt.length);
  }
}

/**
 * Convert an AST back into a prompt string.
 * @param ast ASTNode[]
 * @returns string
 */
export function serialize(ast: ASTNode[]): string {
  const buf: SerializeBuffer = { prompt: '' };
  serializeCore(ast, undefined, buf);
  return buf.prompt;
}

/**
 * Serialize an AST to a prompt string while simultaneously computing the
 * selection range from `isSelection` flags on nodes.
 *
 * This is more reliable than separate serialize + selection computation because
 * the position tracking is guaranteed to match the serialized output.
 */
export function serializeWithSelection(ast: ASTNode[]): {
  prompt: string;
  selectionStart: number;
  selectionEnd: number;
} {
  let selStart = Infinity;
  let selEnd = -1;

  const buf: SerializeBuffer = { prompt: '' };
  serializeCore(
    ast,
    {
      onNode(node, start, end) {
        if (node.isSelection) {
          selStart = Math.min(selStart, start);
          selEnd = Math.max(selEnd, end);
        }
      },
    },
    buf
  );

  if (selStart === Infinity) {
    selStart = 0;
    selEnd = buf.prompt.length;
  }

  return { prompt: buf.prompt, selectionStart: selStart, selectionEnd: selEnd };
}
