import { useCallback, useRef } from 'react';

import { isOtherTextEntry } from './promptFocus';

/** What a prompt field lets its actions do with its prompt. */
export interface PromptEditor {
  /** The prompt as last edited, for code that runs after an await and must not act on a stale render. */
  read: () => string;
  /**
   * Replaces the whole prompt as an edit, so Ctrl+Z in the field undoes it where the browser allows. `withoutFocus`
   * writes the value without taking focus, for a result that lands while the user is elsewhere; Ctrl+Z cannot undo
   * that, so the caller offers an Undo.
   */
  write: (prompt: string, options?: { withoutFocus?: boolean }) => void;
  /** The prompt's textarea, where focus returns once a popover that wrote the prompt closes. */
  element: () => HTMLTextAreaElement | null;
  /** Whether this field is mounted and showing `projectId`'s prompt, so a late result or an Undo may write to it. */
  isShowing: (projectId: string) => boolean;
}

/**
 * Hands focus, when a popover closes after writing the prompt, to the prompt (where Ctrl+Z undoes the write), or to
 * the field the user is typing in, rather than back to the popover's trigger. Pass `finalFocusEl` to the popover and
 * call `keepFocusAfterWrite` just before closing it.
 */
export const useFocusAfterPromptWrite = (editor: Pick<PromptEditor, 'element'>) => {
  const targetRef = useRef<HTMLElement | null>(null);
  const finalFocusEl = useCallback(() => {
    const target = targetRef.current;

    targetRef.current = null;

    return target?.isConnected ? target : null;
  }, []);
  const keepFocusAfterWrite = useCallback(() => {
    const prompt = editor.element();
    const active = document.activeElement;

    targetRef.current = isOtherTextEntry(active, prompt) ? active : prompt;
  }, [editor]);

  return { finalFocusEl, keepFocusAfterWrite };
};
