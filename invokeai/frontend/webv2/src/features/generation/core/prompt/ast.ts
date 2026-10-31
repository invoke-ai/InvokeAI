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
  | { type: 'weight'; value: PromptAttention; range: PromptRange; source?: string }
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

const WORD_CHAR = /[\p{L}\p{N}\p{M}_]/u;
const WHITESPACE = /\s/;
const NUMERIC_WEIGHT = /^[+-]?\d+(?:\.\d+)?/;
const ANIMA_NUMERIC_WEIGHT = /(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)/y;
const SYMBOLIC_WEIGHT = /^(?:\++|-+)/;
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

  if (isWordChar(prompt[end]) || prompt[end] === '+' || prompt[end] === '-') {
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

/** Source ranges are shared by Anima painting, validation and source-preserving attention edits. */
export interface AnimaPart {
  range: PromptRange;
  contentRange: PromptRange;
  text?: string;
  /** Defer copying invalid source until enclosing-group rollback has finished. */
  literalSource?: boolean;
  children: AnimaPart[];
  attention?: PromptAttention;
  attentionRange?: PromptRange;
  weight: number;
  markup: boolean;
  /** Literal call punctuation and authored whitespace keep their source boundaries. */
  glue: boolean;
  delimiter?: '(' | '"';
}

export interface AnimaPrompt {
  nodes: AnimaPart[];
  tokens: PromptToken[];
  text: string;
  ranges: { start: number; end: number; weight: number }[];
}

const ANIMA_SPACE = ' \t\r\n\v\f';
const ANIMA_SYNTAX = '(),.+=-"';
const ANIMA_QUOTED_WHITESPACE: Readonly<Record<string, string>> = { n: '\n', t: '\t', r: '\r', f: '\f' };
// Compel uses Python's str.strip() only for the blank quoted-fragment check.
const ANIMA_QUOTED_BLANK = new Set(
  ' \t\r\n\v\f\u001c\u001d\u001e\u001f\u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'
);
const ANIMA_METHOD = /\.[A-Za-z_][A-Za-z0-9_]*[ \t\r\n\v\f]*\(/y;
export const ANIMA_MAX_WEIGHT = 65504;

const animaMethod = (prompt: string, start: number): RegExpExecArray | null => {
  ANIMA_METHOD.lastIndex = start;
  return ANIMA_METHOD.exec(prompt);
};
const animaSpace = (char: string | undefined): boolean => char !== undefined && ANIMA_SPACE.includes(char);
const animaLiteral = (value: string): string => value.replace(/\\([()"])/g, '$1');

const animaStructures = (prompt: string) => {
  const pairs = new Map<number, number>();
  const rejectedQuotes = new Map<number, number>();
  const stack: number[] = [];
  let quote: number | undefined;
  let quoteInvalid = false;
  let quoteMultiline = false;
  let quoteEscapes: [number, string][] = [];
  const whitespaceEscapes = new Map<number, string>();
  for (let index = 0; index < prompt.length; index++) {
    const char = prompt[index]!;
    if (quote !== undefined && (char === '\v' || char === '\f')) {
      quoteInvalid = true;
    }
    if (char === '\\' && quote !== undefined && index + 1 < prompt.length) {
      const escaped = prompt[index + 1]!;
      if (escaped === 'f') {
        // Compel accepts a blank quoted fragment, but rejects form feed among words.
        quoteInvalid = true;
      }
      const whitespace = ANIMA_QUOTED_WHITESPACE[escaped];
      if (whitespace !== undefined) {
        quoteEscapes.push([index, whitespace]);
        index++;
        continue;
      }
    }
    if (char === '\\' && ANIMA_SYNTAX.includes(prompt[index + 1] ?? '\0')) {
      index++;
    } else if (char === '"') {
      if (quote === undefined) {
        quote = index;
        quoteInvalid = false;
        quoteMultiline = false;
        quoteEscapes = [];
        stack.push(index);
      } else {
        if (
          stack.at(-1) === quote &&
          !quoteMultiline &&
          (!quoteInvalid ||
            Array.from(animaDecodedText(prompt, quote + 1, index, new Map(quoteEscapes))).every((char) =>
              ANIMA_QUOTED_BLANK.has(char)
            ))
        ) {
          pairs.set(stack.pop()!, index);
          for (const [start, value] of quoteEscapes) {
            whitespaceEscapes.set(start, value);
          }
        } else {
          rejectedQuotes.set(quote, index);
          while (stack.length && stack.at(-1) !== quote) {
            stack.pop();
          }
          stack.pop();
        }
        quote = undefined;
      }
    } else if (char === '(') {
      stack.push(index);
    } else if (char === ')' && prompt[stack.at(-1) ?? -1] === '(') {
      pairs.set(stack.pop()!, index);
    } else if ((char === '\r' || char === '\n') && quote !== undefined) {
      // Keep the physical closing boundary even when the quoted target is rejected.
      quoteMultiline = true;
    }
  }
  return { pairs, whitespaceEscapes, rejectedQuotes };
};

const animaSuffix = (
  prompt: string,
  start: number,
  numeric: boolean
): { attention?: PromptAttention; attentionRange?: PromptRange; weight: number; end: number } => {
  const marker = prompt[start];
  if (marker === '+' || marker === '-') {
    let end = start + 1;
    while (prompt[end] === marker) {
      end++;
    }
    return {
      attention: prompt.slice(start, end),
      attentionRange: range(start, end),
      weight: (marker === '+' ? 1.1 : 0.9) ** (end - start),
      end,
    };
  }
  ANIMA_NUMERIC_WEIGHT.lastIndex = start;
  const match = numeric ? ANIMA_NUMERIC_WEIGHT.exec(prompt) : null;
  if (!match) {
    return { weight: 1, end: start };
  }
  let end = start + match[0].length;
  if (prompt[end - 1] === '.' && animaMethod(prompt, end - 1)) {
    end--;
  }
  const weight = Number(prompt.slice(start, end));
  return { attention: weight, attentionRange: range(start, end), weight, end };
};

const animaAttention = (prompt: string, start: number, stop: number, numeric: boolean) => {
  const implicit = animaSuffix(prompt, start, numeric);
  let explicit = start;
  while (explicit < stop && animaSpace(prompt[explicit])) {
    explicit++;
  }
  const method = animaMethod(prompt, explicit);
  if (!method || method[0].split('(')[0]?.trim() !== '.attend') {
    return implicit;
  }
  let value = explicit + method[0].length;
  while (value < stop && animaSpace(prompt[value])) {
    value++;
  }
  const suffix = animaSuffix(prompt, value, true);
  let close = suffix.end;
  while (close < stop && animaSpace(prompt[close])) {
    close++;
  }
  return suffix.end > value && close < stop && prompt[close] === ')' ? { ...suffix, end: close + 1 } : implicit;
};

const animaPart = (start: number, end: number, text?: string, glue = false): AnimaPart => ({
  range: range(start, end),
  contentRange: range(start, end),
  text,
  children: [],
  weight: 1,
  markup: false,
  glue,
});

const makeAnimaLiteral = (part: AnimaPart): void => {
  part.literalSource = true;
  part.children = [];
  part.weight = 1;
  part.attention = undefined;
  part.attentionRange = undefined;
  part.markup = false;
  part.delimiter = undefined;
  part.contentRange = { ...part.range };
};

/** Padded whitespace lets grammar matching retain the authored source offsets. */
const animaSyntaxSource = (prompt: string, whitespaceEscapes: ReadonlyMap<number, string>): string => {
  if (!whitespaceEscapes.size) {
    return prompt;
  }
  const parts: string[] = [];
  let cursor = 0;
  for (const [start, value] of whitespaceEscapes) {
    parts.push(prompt.slice(cursor, start), value, ' ');
    cursor = start + 2;
  }
  parts.push(prompt.slice(cursor));
  return parts.join('');
};

const animaDecodedText = (
  prompt: string,
  start: number,
  end: number,
  whitespaceEscapes: ReadonlyMap<number, string>
): string => {
  if (!whitespaceEscapes.size) {
    return prompt.slice(start, end);
  }
  const parts: string[] = [];
  while (start < end) {
    const value = whitespaceEscapes.get(start);
    parts.push(value ?? prompt[start]!);
    start += value === undefined ? 1 : 2;
  }
  return parts.join('');
};

/** Literal fallback retains comma and method boundaries, including after rejected closed quotes. */
const animaFallbackEnd = (prompt: string, start: number, index: number, stop: number): number => {
  if (index > start && prompt[index - 1] === ',') {
    return index;
  }
  while (index < stop) {
    const char = prompt[index]!;
    if (animaSpace(char) || (index > start && animaMethod(prompt, index))) {
      break;
    }
    index++;
    if (char === ',') {
      break;
    }
  }
  return Math.max(start + 1, index);
};

export const parseAnimaPrompt = (prompt: string): AnimaPrompt => {
  const { pairs, whitespaceEscapes, rejectedQuotes } = animaStructures(prompt);
  let decodedSyntax: string | undefined;
  const root = animaPart(0, prompt.length);
  const frames = [{ parent: root, stop: prompt.length, grouped: false, call: false, quoted: false }];
  let index = 0;
  while (frames.length) {
    const { parent, stop, grouped, call, quoted } = frames.at(-1)!;
    // Decode escapes only within an accepted quoted target.
    const syntax = quoted ? (decodedSyntax ??= animaSyntaxSource(prompt, whitespaceEscapes)) : prompt;
    if (index >= stop) {
      frames.pop();
      if (!frames.length) {
        break;
      }
      if (call && index < prompt.length) {
        parent.children.push(animaPart(index, index + 1, prompt[index], true));
      }
      if (parent.literalSource) {
        const enclosing = frames.at(-1)!;
        if (enclosing.grouped && !enclosing.call) {
          makeAnimaLiteral(enclosing.parent);
          index = enclosing.stop;
          continue;
        }
        if (prompt[parent.range.start] === '"') {
          parent.range.end = animaFallbackEnd(prompt, parent.range.start, stop + 1, enclosing.stop);
          parent.contentRange = { ...parent.range };
        }
      }
      index = parent.range.end;
      continue;
    }
    const char = syntax[index]!;
    if (animaSpace(char)) {
      let end = index + 1;
      while (end < stop && animaSpace(syntax[end])) {
        end++;
      }
      parent.children.push(animaPart(index, end, animaDecodedText(prompt, index, end, whitespaceEscapes), true));
      index = end;
      continue;
    }
    const method = animaMethod(syntax, index);
    if (method) {
      const opening = index + method[0].length - 1;
      const closing = pairs.get(opening) ?? stop;
      const node = animaPart(index, Math.min(closing + 1, stop));
      node.children.push(
        animaPart(index, opening + 1, animaDecodedText(prompt, index, opening + 1, whitespaceEscapes), true)
      );
      parent.children.push(node);
      frames.push({ parent: node, stop: closing, grouped: true, call: true, quoted });
      index = opening + 1;
      continue;
    }
    if (char === '(' || char === '"') {
      const closing = pairs.get(index);
      if (closing === undefined || closing >= stop) {
        if (grouped && !call) {
          makeAnimaLiteral(parent);
          index = stop;
          continue;
        }
        const rejectedClose = rejectedQuotes.get(index);
        if (rejectedClose !== undefined && rejectedClose < stop) {
          const end = animaFallbackEnd(prompt, index, rejectedClose + 1, stop);
          const literal = animaPart(index, end);
          makeAnimaLiteral(literal);
          parent.children.push(literal);
          index = end;
        } else {
          parent.children.push(animaPart(index, stop, prompt.slice(index, stop)));
          index = stop;
        }
        continue;
      }
      const suffix = animaAttention(syntax, closing + 1, stop, true);
      const node = animaPart(index, suffix.end);
      Object.assign(node, {
        attention: suffix.attention,
        attentionRange: suffix.attentionRange,
        weight: suffix.weight,
        markup: true,
        delimiter: char,
      });
      node.contentRange = range(index + 1, closing);
      parent.children.push(node);
      frames.push({ parent: node, stop: closing, grouped: true, call: false, quoted: quoted || char === '"' });
      index++;
      continue;
    }
    if (grouped && (char === ',' || char === '.')) {
      parent.children.push(animaPart(index, index + 1, char));
      index++;
      continue;
    }
    const start = index;
    while (index < stop) {
      const char = syntax[index]!;
      if (char === '\\' && index + 1 < stop && ANIMA_SYNTAX.includes(syntax[index + 1]!)) {
        index += 2;
        if (syntax[index - 1] === ',') {
          break;
        }
      } else if (char === '+' || char === '-') {
        if (index + 1 < stop && !(ANIMA_SPACE + ANIMA_SYNTAX).includes(syntax[index + 1]!)) {
          index++;
        } else {
          break;
        }
      } else if ((ANIMA_SPACE + ANIMA_SYNTAX).includes(char)) {
        break;
      } else {
        index++;
      }
    }
    const targetEnd = index;
    const suffix = targetEnd > start ? animaAttention(syntax, index, stop, false) : { weight: 1, end: index };
    if (suffix.end > targetEnd) {
      const node = animaPart(start, suffix.end, animaLiteral(prompt.slice(start, targetEnd)));
      Object.assign(node, {
        attention: suffix.attention,
        attentionRange: suffix.attentionRange,
        weight: suffix.weight,
        markup: true,
      });
      node.contentRange.end = targetEnd;
      parent.children.push(node);
      index = suffix.end;
      continue;
    }
    if (grouped && targetEnd > start) {
      parent.children.push(animaPart(start, targetEnd, animaLiteral(prompt.slice(start, targetEnd))));
      continue;
    }
    if (grouped && !call) {
      makeAnimaLiteral(parent);
      index = stop;
      continue;
    }
    index = animaFallbackEnd(syntax, start, Math.max(start, targetEnd), stop);
    parent.children.push(animaPart(start, index, animaLiteral(prompt.slice(start, index))));
  }

  const tokens: PromptToken[] = [];
  const ranges: AnimaPrompt['ranges'] = [];
  const text: string[] = [];
  let length = 0;
  let markup = false;
  let previousGlue = true;
  const tasks = root.children.map((part) => ({ part, weight: 1 })).reverse();
  while (tasks.length) {
    const { part, weight: inherited } = tasks.pop()!;
    const weight = inherited * part.weight;
    if (!Number.isFinite(weight) || Math.abs(weight) > ANIMA_MAX_WEIGHT || part.weight > ANIMA_MAX_WEIGHT) {
      throw new TypeError('Anima prompt weight must be finite and within float16 magnitude.');
    }
    markup ||= part.markup;
    if (part.literalSource) {
      part.text = prompt.slice(part.range.start, part.range.end);
    }
    if (part.text === undefined) {
      for (let child = part.children.length - 1; child >= 0; child--) {
        tasks.push({ part: part.children[child]!, weight });
      }
    } else if (part.text) {
      if (text.length && !previousGlue && !part.glue && !animaSpace(text.at(-1)?.at(-1)) && !animaSpace(part.text[0])) {
        text.push(' ');
        ranges.push({ start: length, end: ++length, weight: 1 });
      }
      text.push(part.text);
      ranges.push({ start: length, end: length + part.text.length, weight });
      length += part.text.length;
      previousGlue = part.glue;
    }
  }
  // Paint only syntax accepted by the same parse. Invalid enclosing groups have no syntax children.
  const painting = [...root.children].reverse();
  while (painting.length) {
    const part = painting.pop()!;
    if (part.delimiter) {
      tokens.push({ type: 'lparen', range: range(part.range.start, part.contentRange.start) });
      tokens.push({ type: 'rparen', range: range(part.contentRange.end, part.contentRange.end + 1) });
    }
    if (part.attention !== undefined) {
      const tailStart = part.contentRange.end + (part.delimiter ? 1 : 0);
      const attentionRange = part.attentionRange ?? range(tailStart, part.range.end);
      if (tailStart < attentionRange.start) {
        tokens.push({
          type: 'word',
          value: prompt.slice(tailStart, attentionRange.start),
          range: range(tailStart, attentionRange.start),
        });
      }
      tokens.push({ type: 'weight', value: part.attention, range: attentionRange });
      if (attentionRange.end < part.range.end) {
        tokens.push({
          type: 'word',
          value: prompt.slice(attentionRange.end, part.range.end),
          range: range(attentionRange.end, part.range.end),
        });
      }
    }
    if (part.text !== undefined) {
      tokens.push({
        type: part.glue && /^[ \t\r\n\v\f]+$/.test(part.text) ? 'whitespace' : 'word',
        value: part.text,
        range: { ...part.contentRange },
      });
    }
    for (let child = part.children.length - 1; child >= 0; child--) {
      painting.push(part.children[child]!);
    }
  }
  tokens.sort((a, b) => a.range.start - b.range.start);
  return {
    nodes: root.children,
    tokens,
    text: !markup && !/\\[()"]/.test(prompt) ? prompt : text.join(''),
    ranges: !markup && !/\\[()"]/.test(prompt) ? (prompt ? [{ start: 0, end: prompt.length, weight: 1 }] : []) : ranges,
  };
};

const readWord = (prompt: string, start: number): { token: PromptToken; extraToken?: PromptToken; end: number } => {
  const first = String.fromCodePoint(prompt.codePointAt(start) ?? 0);
  let end = start + first.length;

  while (isWordChar(first) && end < prompt.length) {
    const next = String.fromCodePoint(prompt.codePointAt(end) ?? 0);
    if (!isWordChar(next)) {
      break;
    }
    end += next.length;
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

export const tokenizePrompt = (prompt: string, syntaxProfile: 'compel' | 'anima' = 'compel'): PromptToken[] => {
  if (syntaxProfile === 'anima') {
    return parseAnimaPrompt(prompt).tokens;
  }
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

      const weight = readNumericAttention(prompt, index + 1) ?? readSymbolicAttention(prompt, index + 1);

      if (weight) {
        tokens.push({
          type: 'weight',
          value: weight.value,
          range: range(index + 1, weight.end),
          source: prompt.slice(index + 1, weight.end),
        });
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

    if (!PUNCTUATION.has(char)) {
      const result = readWord(prompt, index);
      tokens.push(result.token);

      if (result.extraToken) {
        tokens.push(result.extraToken);
      }

      index = result.end;
      continue;
    }

    tokens.push({ type: 'punct', value: char, range: range(index, index + 1) });
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
      return token.source ?? String(token.value);
    default:
      return token.value;
  }
};

class PromptParser {
  private index = 0;
  private lastEmbeddingEnd = -1;
  private readonly groupEnds = new Map<number, number>();

  constructor(private readonly tokens: PromptToken[]) {
    const openings: number[] = [];
    for (let index = 0; index < tokens.length; index++) {
      const type = tokens[index]?.type;
      if (type === 'rembed') {
        this.lastEmbeddingEnd = index;
      } else if (type === 'lparen') {
        openings.push(index);
      } else if (type === 'rparen') {
        const opening = openings.pop();
        if (opening !== undefined) {
          this.groupEnds.set(opening, index);
        }
      }
    }
  }

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
    const closing = this.groupEnds.get(this.index - 1);
    if (closing === undefined) {
      return false;
    }
    let tailOffset = closing - this.index + 1;
    while (this.peek(tailOffset)?.type === 'whitespace') {
      tailOffset++;
    }
    if (
      !this.isPunct(tailOffset, '.') ||
      this.peek(tailOffset + 1)?.type !== 'word' ||
      this.peek(tailOffset + 2)?.type !== 'lparen'
    ) {
      return false;
    }
    let offset = 0;
    let depth = 0;
    let hasComma = false;

    while (this.peek(offset)) {
      const token = this.peek(offset);

      if (token?.type === 'lparen') {
        depth++;
      } else if (token?.type === 'rparen') {
        if (depth === 0) {
          return hasComma;
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

  private parseNodes(inGroup = false): PromptAstNode[] {
    const nodes: PromptAstNode[] = [];

    while (this.peek()) {
      if (inGroup && this.peek()?.type === 'rparen') {
        break;
      }
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

          const children = this.parseNodes(true);
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
          } else {
            nodes.push({ type: 'punct', value: '(', range: lparen.range }, ...children);
            break;
          }

          nodes.push({ type: 'group', attention, children, range: range(lparen.range.start, nodeEnd) });
          break;
        }
        case 'lembed': {
          if (this.lastEmbeddingEnd < this.index) {
            nodes.push({ type: 'punct', value: '<', range: token.range });
            break;
          }
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

          nodes.push({ type: 'embedding', value, range: range(token.range.start, nodeEnd) });
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
          nodes.push({ type: 'punct', value: tokenSource(token), range: token.range });
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

export const parsePrompt = (prompt: string, syntaxProfile: 'compel' | 'anima' = 'compel'): PromptAstNode[] => {
  if (syntaxProfile === 'compel') {
    return parsePromptTokens(tokenizePrompt(prompt));
  }
  const result: PromptAstNode[] = [];
  const tasks = parseAnimaPrompt(prompt)
    .nodes.map((part) => ({ part, destination: result }))
    .reverse();
  while (tasks.length) {
    const { part, destination } = tasks.pop()!;
    if (part.delimiter) {
      const children: PromptAstNode[] = [];
      destination.push({ type: 'group', children, attention: part.attention, range: part.range });
      for (let child = part.children.length - 1; child >= 0; child--) {
        tasks.push({ part: part.children[child]!, destination: children });
      }
    } else if (part.text !== undefined) {
      destination.push(
        part.glue && /^[ \t\r\n\v\f]+$/.test(part.text)
          ? { type: 'whitespace', value: part.text, range: part.range }
          : { type: 'word', text: part.text, attention: part.attention, range: part.range }
      );
    } else {
      for (let child = part.children.length - 1; child >= 0; child--) {
        tasks.push({ part: part.children[child]!, destination });
      }
    }
  }
  return result;
};

interface SerializeVisitor {
  onNode?: (node: PromptAstNode, start: number, end: number) => void;
}

export const serializePromptAttention = (attention: PromptAttention | undefined): string => {
  const value = attention === undefined ? '' : String(attention);
  if (typeof attention !== 'number' || !value.includes('e')) {
    return value;
  }
  // Compel's numeric syntax does not accept exponent notation.
  const [coefficient = '', exponent = '0'] = value.split('e');
  const sign = coefficient.startsWith('-') ? '-' : '';
  const [whole = '', fraction = ''] = coefficient.replace('-', '').split('.');
  const digits = whole + fraction;
  const point = whole.length + Number(exponent);
  if (point <= 0) {
    return `${sign}0.${'0'.repeat(-point)}${digits}`;
  }
  return point >= digits.length
    ? `${sign}${digits}${'0'.repeat(point - digits.length)}`
    : `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
};

const needsAttentionBoundary = (node: PromptAstNode, next: PromptAstNode | undefined): boolean => {
  if ((node.type !== 'word' && node.type !== 'group') || node.attention === undefined || !next) {
    return false;
  }
  const nextText = next.type === 'word' ? next.text : next.type === 'punct' ? next.value : '';
  return (
    isWordChar(nextText[0]) || (typeof node.attention === 'number' ? nextText.startsWith('.') : /^[+-]/.test(nextText))
  );
};

const serializeInto = (
  nodes: PromptAstNode[],
  output: { value: string },
  visitor?: SerializeVisitor,
  source?: string
): void => {
  for (let index = 0; index < nodes.length; index++) {
    const node = nodes[index]!;
    const start = output.value.length;
    const needsBoundary = needsAttentionBoundary(node, nodes[index + 1]);
    if (needsBoundary) {
      output.value += '(';
    }

    switch (node.type) {
      case 'punct':
      case 'whitespace':
        output.value += node.value;
        break;
      case 'escaped_paren':
        output.value += `\\${node.value}`;
        break;
      case 'word':
        output.value += `${node.text}${serializePromptAttention(node.attention)}`;
        break;
      case 'group':
        output.value += '(';
        serializeInto(node.children, output, visitor, source);
        output.value += `)${serializePromptAttention(node.attention)}`;
        break;
      case 'embedding':
        output.value += `<${node.value}>`;
        break;
      case 'prompt_function':
        if (source !== undefined) {
          let cursor = node.range.start;
          for (const arg of node.promptArgs) {
            output.value += source.slice(cursor, arg.contentRange.start);
            serializeInto(arg.nodes, output, visitor, source);
            cursor = arg.contentRange.end;
          }
          output.value += source.slice(cursor, node.range.end);
          break;
        }
        output.value += '(';
        node.promptArgs.forEach((arg, index) => {
          if (index > 0) {
            output.value += `,${arg.separator ?? ' '}`;
          }

          output.value += arg.quote;
          serializeInto(arg.nodes, output, visitor, source);
          output.value += CLOSE_QUOTE_BY_OPEN[arg.quote] ?? arg.quote;
        });
        output.value += `).${node.name}(${node.functionParams})`;
        break;
    }

    if (needsBoundary) {
      output.value += ')';
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
  nodes: PromptAstNode[],
  source?: string
): { prompt: string; selectionStart: number; selectionEnd: number } => {
  const output = { value: '' };
  let selectionStart = Infinity;
  let selectionEnd = -1;

  serializeInto(
    nodes,
    output,
    {
      onNode: (node, start, end) => {
        if (!node.isSelection) {
          return;
        }

        selectionStart = Math.min(selectionStart, start);
        selectionEnd = Math.max(selectionEnd, end);
      },
    },
    source
  );

  if (selectionStart === Infinity) {
    return { prompt: output.value, selectionStart: 0, selectionEnd: output.value.length };
  }

  return { prompt: output.value, selectionStart, selectionEnd };
};
