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
 */
export const usePromptAttentionHotkeys = ({ textareaRef, onPromptChange }: UsePromptAttentionHotkeysArgs) => {
  const isPromptFocused = useCallback(() => document.activeElement === textareaRef.current, [textareaRef]);

  const handleAttentionAdjustment = useCallback(
    (direction: 'increment' | 'decrement') => {
      const textarea = textareaRef.current;
      if (!textarea) {
        return;
      }

      const result = adjustPromptAttention(textarea.value, textarea.selectionStart, textarea.selectionEnd, direction);

      onPromptChange(result.prompt);

      // Update selection after React re-renders
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.setSelectionRange(result.selectionStart, result.selectionEnd);
          textareaRef.current.focus();
        }
      }, 0);
    },
    [textareaRef, onPromptChange]
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
