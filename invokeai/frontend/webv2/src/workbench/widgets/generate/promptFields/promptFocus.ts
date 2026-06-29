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
