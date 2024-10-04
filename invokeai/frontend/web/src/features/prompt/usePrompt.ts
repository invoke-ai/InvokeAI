import { useDisclosure } from '@invoke-ai/ui-library';
import { isNil } from 'lodash-es';
import type { ChangeEventHandler, KeyboardEventHandler, RefObject } from 'react';
import { useCallback } from 'react';
import { flushSync } from 'react-dom';

type UseInsertTriggerArg = {
  prompt: string;
  textareaRef: RefObject<HTMLTextAreaElement>;
  onChange: (v: string) => void;
};

export const usePrompt = ({ prompt, textareaRef, onChange: _onChange }: UseInsertTriggerArg) => {
  const { isOpen, onClose, onOpen } = useDisclosure();

  const onChange: ChangeEventHandler<HTMLTextAreaElement> = useCallback(
    (e) => {
      _onChange(e.target.value);
    },
    [_onChange]
  );

  const insertTrigger = useCallback(
    (v: string) => {
      if (!textareaRef.current) {
        return;
      }

      // this is where we insert the trigger
      const caret = textareaRef.current.selectionStart;

      if (isNil(caret)) {
        return;
      }

      let newPrompt = prompt.slice(0, caret);

      newPrompt += `${v}`;

      // we insert the cursor after the end of trigger
      const finalCaretPos = newPrompt.length;

      newPrompt += prompt.slice(caret);

      // must flush dom updates else selection gets reset
      flushSync(() => {
        _onChange(newPrompt);
      });

      // set the cursor position to just after the trigger
      textareaRef.current.selectionStart = finalCaretPos;
      textareaRef.current.selectionEnd = finalCaretPos;
    },
    [textareaRef, _onChange, prompt]
  );

  const onFocus = useCallback(() => {
    textareaRef.current?.focus();
  }, [textareaRef]);

  const handleClosePopover = useCallback(() => {
    onClose();
    onFocus();
  }, [onFocus, onClose]);

  const onSelect = useCallback(
    (v: string) => {
      insertTrigger(v);
      handleClosePopover();
    },
    [handleClosePopover, insertTrigger]
  );

  const onKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = useCallback(
    (e) => {
      if (e.key === '<') {
        onOpen();
        e.preventDefault();
      }
    },
    [onOpen]
  );

  return {
    onChange,
    isOpen,
    onClose,
    onOpen,
    onSelect,
    onKeyDown,
    onFocus,
  };
};
