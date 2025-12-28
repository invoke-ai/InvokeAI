import { adjustPromptAttention } from 'common/util/promptAttention';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { RefObject } from 'react';
import { useCallback } from 'react';

type UsePromptAttentionHotkeysArgs = {
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  onPromptChange: (prompt: string) => void;
};

/**
 * Registers hotkeys for adjusting prompt attention weights (Ctrl+Up/Down).
 * Handles both increment and decrement operations with proper selection restoration.
 * Uses execCommand to integrate with browser's native undo stack.
 */
export const usePromptAttentionHotkeys = ({
  textareaRef,
  onPromptChange: _onPromptChange,
}: UsePromptAttentionHotkeysArgs) => {
  const isPromptFocused = useCallback(() => document.activeElement === textareaRef.current, [textareaRef]);

  const handleAttentionAdjustment = useCallback(
    (direction: 'increment' | 'decrement') => {
      const textarea = textareaRef.current;
      if (!textarea) {
        return;
      }

      const result = adjustPromptAttention(textarea.value, textarea.selectionStart, textarea.selectionEnd, direction);

      // Use execCommand to make the change undo-able by the browser.
      // This triggers the textarea's native onChange, which syncs React state.
      textarea.focus();
      textarea.setSelectionRange(0, textarea.value.length);
      document.execCommand('insertText', false, result.prompt);

      // Restore the selection to cover the adjusted portion
      textarea.setSelectionRange(result.selectionStart, result.selectionEnd);
    },
    [textareaRef]
  );

  useRegisteredHotkeys({
    id: 'promptWeightUp',
    category: 'app',
    callback: (e) => {
      if (isPromptFocused() && textareaRef.current) {
        e.preventDefault();
        handleAttentionAdjustment('increment');
      }
    },
    options: { preventDefault: true, enableOnFormTags: ['TEXTAREA'] },
    dependencies: [isPromptFocused, handleAttentionAdjustment],
  });

  useRegisteredHotkeys({
    id: 'promptWeightDown',
    category: 'app',
    callback: (e) => {
      if (isPromptFocused() && textareaRef.current) {
        e.preventDefault();
        handleAttentionAdjustment('decrement');
      }
    },
    options: { preventDefault: true, enableOnFormTags: ['TEXTAREA'] },
    dependencies: [isPromptFocused, handleAttentionAdjustment],
  });
};
