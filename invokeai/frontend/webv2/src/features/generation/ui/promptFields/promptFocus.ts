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

export interface PromptTriggerMatch {
  /** The keystroke the picker consumed, so a dismissal can put it back. */
  key: PromptTriggerKey;
  /** The span the picked trigger replaces, including anything already typed. */
  range: PromptTextRange;
}

/** `_` is a word character, so `snake__case` must not read as the start of a reference. */
const isWordCharacter = (character: string | undefined): boolean =>
  character !== undefined && /[A-Za-z0-9_]/.test(character);

/**
 * Whether `key` opens the trigger picker, and what it would replace.
 *
 * `<` opens outright — it only ever starts an embedding token. `__` opens on the
 * second underscore, but only where a wildcard reference could actually begin:
 * after whitespace, punctuation, or the start of the prompt. That leaves
 * `snake__case` as ordinary text, which the same rule in
 * `core/dynamicPrompts.ts` already excludes from being a reference.
 */
export const getPromptTriggerRange = (
  value: string,
  selectionStart: number,
  selectionEnd: number,
  key: string,
  keys: readonly PromptTriggerKey[]
): PromptTriggerMatch | null => {
  if (key === '<' && keys.includes('<')) {
    return { key: '<', range: { end: selectionEnd, start: selectionStart } };
  }

  if (key === '_' && keys.includes('_') && value[selectionStart - 1] === '_') {
    return isWordCharacter(value[selectionStart - 2])
      ? null
      : // The underscore already typed is part of the reference, so the picked
        // `__name__` replaces it rather than landing after it.
        { key: '_', range: { end: selectionEnd, start: selectionStart - 1 } };
  }

  return null;
};
