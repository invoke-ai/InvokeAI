import { useDisclosure } from '@invoke-ai/ui-library';
import { isNil } from 'lodash-es';
import type { ChangeEventHandler, KeyboardEventHandler, RefObject } from 'react';
import { useCallback } from 'react';
import { flushSync } from 'react-dom';

export type UseInsertEmbeddingArg = {
  prompt: string;
  textareaRef: RefObject<HTMLTextAreaElement>;
  onChange: (v: string) => void;
};

export const usePrompt = ({ prompt, textareaRef, onChange: _onChange }: UseInsertEmbeddingArg) => {
  const { isOpen, onClose, onOpen } = useDisclosure();

  const onChange: ChangeEventHandler<HTMLTextAreaElement> = useCallback(
    (e) => {
      _onChange(e.target.value);
    },
    [_onChange]
  );

  const insertEmbedding = useCallback(
    (v: string) => {
      if (!textareaRef.current) {
        return;
      }

      // this is where we insert the TI trigger
      const caret = textareaRef.current.selectionStart;

      if (isNil(caret)) {
        return;
      }

      let newPrompt = prompt.slice(0, caret);

      if (newPrompt[newPrompt.length - 1] !== '<') {
        newPrompt += '<';
      }

      newPrompt += `${v}>`;

      // we insert the cursor after the `>`
      const finalCaretPos = newPrompt.length;

      newPrompt += prompt.slice(caret);

      // must flush dom updates else selection gets reset
      flushSync(() => {
        _onChange(newPrompt);
      });

      // set the caret position to just after the TI trigger
      textareaRef.current.selectionStart = finalCaretPos;
      textareaRef.current.selectionEnd = finalCaretPos;
    },
    [textareaRef, _onChange, prompt]
  );

  const onFocus = useCallback(() => {
    textareaRef.current?.focus();
  }, [textareaRef]);

  const handleClose = useCallback(() => {
    onClose();
    onFocus();
  }, [onFocus, onClose]);

  const onSelectEmbedding = useCallback(
    (v: string) => {
      insertEmbedding(v);
      handleClose();
    },
    [handleClose, insertEmbedding]
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
    onSelectEmbedding,
    onKeyDown,
    onFocus,
  };
};
