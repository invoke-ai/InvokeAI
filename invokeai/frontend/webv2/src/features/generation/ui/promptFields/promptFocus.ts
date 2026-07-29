import { MAX_WILDCARD_NAME_LENGTH } from '@features/generation/core/dynamicPrompts';

let positivePromptElement: HTMLTextAreaElement | null = null;

export type PromptTextRange = { end: number; start: number };

export const registerPositivePromptElement = (element: HTMLTextAreaElement | null): void => {
  positivePromptElement = element;
};

export const focusPositivePrompt = (): boolean => {
  if (!positivePromptElement) {
    return false;
  }

  positivePromptElement.focus();
  positivePromptElement.select();

  return true;
};

export const isPositivePromptFocused = (): boolean =>
  positivePromptElement !== null && document.activeElement === positivePromptElement;

const clamp = (value: number, min: number, max: number): number => Math.min(Math.max(value, min), max);

export const insertTextAtRange = (
  value: string,
  text: string,
  range: PromptTextRange | undefined,
  fallbackCaret = value.length
): { caret: number; value: string } => {
  const insertionRange = range ?? { end: fallbackCaret, start: fallbackCaret };
  const start = clamp(Math.min(insertionRange.start, insertionRange.end), 0, value.length);
  const end = clamp(Math.max(insertionRange.start, insertionRange.end), 0, value.length);
  const nextValue = `${value.slice(0, start)}${text}${value.slice(end)}`;

  return { caret: start + text.length, value: nextValue };
};

export const insertPromptText = ({
  onChange,
  range,
  textarea,
  text,
  value,
}: {
  onChange: (value: string) => void;
  range?: PromptTextRange;
  textarea: HTMLTextAreaElement | null;
  text: string;
  value: string;
}): void => {
  const caret = textarea?.selectionStart ?? value.length;
  const { caret: nextCaret, value: nextValue } = insertTextAtRange(value, text, range, caret);

  onChange(nextValue);

  window.requestAnimationFrame(() => {
    textarea?.focus();
    textarea?.setSelectionRange(nextCaret, nextCaret);
  });
};

export const insertPositivePromptText = ({
  onChange,
  range,
  text,
  value,
}: {
  onChange: (value: string) => void;
  range?: PromptTextRange;
  text: string;
  value: string;
}): void => insertPromptText({ onChange, range, textarea: positivePromptElement, text, value });

/** Which trigger keys a field answers to. Wildcards only exist on the positive prompt. */
export type PromptTriggerKey = '<' | '_';

export interface PromptTriggerQuery {
  key: PromptTriggerKey;
  /** What has been typed since the trigger, for narrowing the list. */
  query: string;
  /** The span a picked option replaces, trigger and typed query included. */
  range: PromptTextRange;
}

/**
 * Past this the user is writing prose, not naming a wildcard, and an
 * autocomplete still hanging around is in the way.
 */
const MAX_TRIGGER_QUERY_LENGTH = MAX_WILDCARD_NAME_LENGTH + 2;

/**
 * The unfinished trigger the caret is sitting inside, or `null`.
 *
 * This replaced an open-on-keystroke picker that swallowed the `__` or `<` and
 * gave the user a separate search box to type into. Here the trigger is typed
 * into the prompt like any other character and whatever follows it narrows the
 * list, so you type where you are already looking — and there is nothing to hand
 * back when the list is dismissed.
 *
 * Scanning stops at a newline, at whitespace, and at anything that closes the
 * token, so a finished `__colours__` or `<embedding>` behind the caret does not
 * re-open the list, and a space ends the attempt rather than leaving it open
 * over a sentence.
 */
export const getActiveTriggerQuery = (
  value: string,
  caret: number,
  keys: readonly PromptTriggerKey[]
): PromptTriggerQuery | null => {
  // At a closing `__`, the backwards scan reaches that pair before its opening
  // pair. Treat a non-empty delimited span as finished even when its content is
  // malformed: autocomplete must not reopen over text the user has already
  // closed.
  const closingStart = caret - 2;
  const openingStart = value.lastIndexOf('__', closingStart - 1);

  if (
    keys.includes('_') &&
    value.slice(closingStart, caret) === '__' &&
    openingStart >= 0 &&
    openingStart + 2 < closingStart
  ) {
    return null;
  }

  const limit = Math.max(0, caret - MAX_TRIGGER_QUERY_LENGTH);

  for (let index = caret - 1; index >= limit; index--) {
    const char = value[index];

    if (char === '\n' || char === '>') {
      return null;
    }

    if (char === '<' && keys.includes('<')) {
      return { key: '<', query: value.slice(index + 1, caret), range: { end: caret, start: index } };
    }

    // A space ends a wildcard attempt but not an embedding one. Embedding model
    // names routinely have spaces in them — `bad hands 5` — so stopping here
    // meant `<bad ` closed the list for good and the name could never be
    // reached inline, even though the filter would have matched it. A wildcard
    // name cannot contain a space, so there the space really is the end.
    if (/\s/.test(char ?? '')) {
      return keys.includes('<') ? findEnclosingEmbedding(value, caret, limit, index) : null;
    }

    if (char === '_' && value[index - 1] === '_' && keys.includes('_')) {
      const query = value.slice(index + 1, caret);

      // `__colours__` with the caret after it: the scan reached the *opening*
      // pair, and the closing one is sitting in the query.
      return query.includes('__') || query.length > MAX_WILDCARD_NAME_LENGTH
        ? null
        : { key: '_', query, range: { end: caret, start: index - 1 } };
    }
  }

  return null;
};

/**
 * Keeps scanning back past spaces for an unclosed `<`, having already met one.
 *
 * Split out so the common path — no space between the caret and the trigger —
 * stays a single loop, and so the two stopping rules read separately: this one
 * only stops at a newline or a `>`, which closes the token.
 */
const findEnclosingEmbedding = (
  value: string,
  caret: number,
  limit: number,
  from: number
): PromptTriggerQuery | null => {
  for (let index = from; index >= limit; index--) {
    const char = value[index];

    if (char === '\n' || char === '>') {
      return null;
    }

    if (char === '<') {
      return { key: '<', query: value.slice(index + 1, caret), range: { end: caret, start: index } };
    }
  }

  return null;
};
