import { useDisclosure } from '@invoke-ai/ui-library';
import { isNil } from 'lodash-es';
import type { ChangeEventHandler, FormEvent, FormEventHandler, KeyboardEventHandler, RefObject, SyntheticEvent } from 'react';
import { useCallback } from 'react';
import { flushSync } from 'react-dom';

type UseInsertTriggerArg = {
  prompt: string;
  paragraphRef: RefObject<HTMLParagraphElement>;
  onChange: (v: string) => void;
};

export const usePromptContentEditable = ({ prompt, paragraphRef, onChange: _onChange }: UseInsertTriggerArg) => {
  const { isOpen, onClose, onOpen } = useDisclosure();


  const onChange = useCallback(
    (e: any) => {
      e.preventDefault();
      _onChange(e.data)
    },
    [_onChange]
  );


  const insertTrigger = useCallback(
    (v: string) => {
      const element = paragraphRef.current;
      if (!element) {
        return;
      }

      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0) {
        // Insert at the end if no selection found
        const newPrompt = prompt + v;
        flushSync(() => {
          _onChange(newPrompt);
        });
        return;
      }

      const range = selection.getRangeAt(0);
      const cursorPosition = range.startOffset;


      console.log({ cursorPosition })


      const updatedPrompt = prompt.slice(0, cursorPosition) + v + prompt.slice(cursorPosition);

      console.log({ updatedPrompt })

      flushSync(() => {
        _onChange(updatedPrompt);
      });

    },
    [paragraphRef, _onChange, prompt]
  );

  const onFocus = useCallback(() => {
    paragraphRef.current?.focus();
  }, [paragraphRef]);

  const handleClosePopover = useCallback(() => {
    onClose();
    onFocus();
  }, [onFocus, onClose]);

  const onSelect = useCallback(
    (v: string) => {
      insertTrigger(v)
      handleClosePopover();
    },
    [handleClosePopover, insertTrigger]
  );

  const onKeyDown: KeyboardEventHandler<HTMLParagraphElement> = useCallback(
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
