import type { PromptAttentionDirection } from '@workbench/generation/prompt/attention';

import { adjustPromptAttention } from '@workbench/generation/prompt/attention';

export const PROMPT_ATTENTION_TARGET_ATTR = 'data-prompt-attention-target';
export const PROMPT_ATTENTION_TARGET_PROPS = { [PROMPT_ATTENTION_TARGET_ATTR]: 'true' } as const;

const isPromptAttentionTarget = (element: Element | null): element is HTMLTextAreaElement =>
  element instanceof HTMLTextAreaElement && element.getAttribute(PROMPT_ATTENTION_TARGET_ATTR) === 'true';

const setTextareaValue = (textarea: HTMLTextAreaElement, value: string): void => {
  const valueSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;

  if (valueSetter) {
    valueSetter.call(textarea, value);
  } else {
    textarea.value = value;
  }

  textarea.dispatchEvent(new Event('input', { bubbles: true }));
};

const replaceTextareaValue = (textarea: HTMLTextAreaElement, value: string): void => {
  textarea.focus();
  textarea.setSelectionRange(0, textarea.value.length);

  if (typeof document.execCommand === 'function' && document.execCommand('insertText', false, value)) {
    return;
  }

  setTextareaValue(textarea, value);
};

export const adjustFocusedPromptAttention = (
  direction: PromptAttentionDirection,
  preferNumericAttentionStyle: boolean
): boolean => {
  const textarea = document.activeElement;

  if (!isPromptAttentionTarget(textarea)) {
    return false;
  }

  const result = adjustPromptAttention(
    textarea.value,
    textarea.selectionStart,
    textarea.selectionEnd,
    direction,
    preferNumericAttentionStyle
  );

  if (result.prompt === textarea.value) {
    return false;
  }

  replaceTextareaValue(textarea, result.prompt);
  textarea.setSelectionRange(result.selectionStart, result.selectionEnd);

  return true;
};
