let positivePromptElement: HTMLTextAreaElement | null = null;

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

export const insertPositivePromptText = ({
  onChange,
  text,
  value,
}: {
  onChange: (value: string) => void;
  text: string;
  value: string;
}): void => {
  const caret = positivePromptElement?.selectionStart ?? value.length;
  const nextValue = `${value.slice(0, caret)}${text}${value.slice(caret)}`;
  const nextCaret = caret + text.length;

  onChange(nextValue);

  window.requestAnimationFrame(() => {
    positivePromptElement?.focus();
    positivePromptElement?.setSelectionRange(nextCaret, nextCaret);
  });
};
