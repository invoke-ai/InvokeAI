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

export type PromptTriggerKey = '<' | '_';

export interface PromptTriggerQuery {
  key: PromptTriggerKey;
  query: string;
  range: PromptTextRange;
}

const MAX_TRIGGER_QUERY_LENGTH = MAX_WILDCARD_NAME_LENGTH + 2;

const isClosingWildcardDelimiter = (value: string, caret: number): boolean => {
  let tokenStart = caret - 2;

  while (tokenStart > 0 && !/[\s>]/.test(value[tokenStart - 1] ?? '')) {
    tokenStart--;
  }

  let delimiterCount = 0;

  for (let index = tokenStart; index < caret;) {
    if (value[index] === '_' && value[index + 1] === '_') {
      delimiterCount++;
      index += 2;
    } else {
      index++;
    }
  }

  return delimiterCount % 2 === 0;
};

export const getActiveTriggerQuery = (
  value: string,
  caret: number,
  keys: readonly PromptTriggerKey[]
): PromptTriggerQuery | null => {
  const closingStart = caret - 2;

  if (keys.includes('_') && value.slice(closingStart, caret) === '__' && isClosingWildcardDelimiter(value, caret)) {
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

    if (/\s/.test(char ?? '')) {
      return keys.includes('<') ? findEnclosingEmbedding(value, caret, limit, index) : null;
    }

    if (char === '_' && value[index - 1] === '_' && keys.includes('_')) {
      const query = value.slice(index + 1, caret);

      return query.includes('__') || query.length > MAX_WILDCARD_NAME_LENGTH
        ? null
        : { key: '_', query, range: { end: caret, start: index - 1 } };
    }
  }

  return null;
};

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
