import type { ChangeEvent, KeyboardEvent, RefObject } from 'react';
import { useCallback, useEffect, useState } from 'react';

type UseEditableArg = {
  value: string;
  defaultValue: string;
  onChange: (value: string) => void;
  onStartEditing?: () => void;
  inputRef?: RefObject<HTMLInputElement | HTMLTextAreaElement>;
};

export const useEditable = ({ value, defaultValue, onChange: _onChange, onStartEditing, inputRef }: UseEditableArg) => {
  const [isEditing, setIsEditing] = useState(false);
  const [localValue, setLocalValue] = useState(value);

  const onBlur = useCallback(() => {
    const trimmedValue = localValue.trim();
    const newValue = trimmedValue || defaultValue;
    setLocalValue(newValue);
    if (newValue !== value) {
      _onChange(newValue);
    }
    setIsEditing(false);
    inputRef?.current?.setSelectionRange(0, 0);
  }, [localValue, defaultValue, value, inputRef, _onChange]);

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setLocalValue(e.target.value);
  }, []);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        onBlur();
      } else if (e.key === 'Escape') {
        setLocalValue(value);
        _onChange(value);
        setIsEditing(false);
      }
    },
    [_onChange, onBlur, value]
  );

  const startEditing = useCallback(() => {
    setIsEditing(true);
    onStartEditing?.();
  }, [onStartEditing]);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalValue(value);
  }, [value]);

  useEffect(() => {
    if (isEditing) {
      inputRef?.current?.focus();
      inputRef?.current?.select();
    }
  }, [inputRef, isEditing]);

  return {
    isEditing,
    startEditing,
    value: localValue,
    inputProps: {
      value: localValue,
      onChange,
      onKeyDown,
      onBlur,
    },
  };
};
