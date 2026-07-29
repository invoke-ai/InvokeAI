import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { CaretRect } from '@features/generation/ui/promptFields/promptCaret';
import type { PromptTriggerKey, PromptTriggerQuery } from '@features/generation/ui/promptFields/promptFocus';
import type { PromptTriggerOption } from '@features/generation/ui/promptFields/promptTriggerOptions';
import type { CompositionEvent, KeyboardEvent, ReactNode } from 'react';

import { getTextareaCaretRect } from '@features/generation/ui/promptFields/promptCaret';
import { getActiveTriggerQuery, insertPromptText } from '@features/generation/ui/promptFields/promptFocus';
import { PromptTriggerAutocomplete } from '@features/generation/ui/promptFields/PromptTriggerAutocomplete';
import {
  getInlineTriggerOptions,
  usePromptTriggerOptions,
} from '@features/generation/ui/promptFields/promptTriggerOptions';
import { DismissOnViewportChange } from '@features/generation/ui/promptFields/useDismissOnViewportChange';
import { useCallback, useId, useMemo, useRef, useState } from 'react';

const CARET_KEYS = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
const LIST_KEYS = ['ArrowUp', 'ArrowDown'];

interface AutocompleteState {
  caretRect: CaretRect;
  query: PromptTriggerQuery;
  textarea: HTMLTextAreaElement;
}

export interface PromptTriggerAutocompleteApi {
  comboboxProps: {
    'aria-activedescendant': string | undefined;
    'aria-autocomplete': 'list';
    'aria-controls': string | undefined;
    'aria-expanded': boolean;
    role: 'combobox';
    onCompositionEnd: (event: CompositionEvent<HTMLTextAreaElement>) => void;
    onCompositionStart: () => void;
    onKeyUp: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  };
  element: ReactNode;
  isOpen: boolean;
  close: () => void;
  refresh: (textarea: HTMLTextAreaElement | null) => void;
  handleKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}

export const usePromptTriggerAutocomplete = ({
  isDisabled = false,
  keys,
  loras,
  onChange,
  selectedModel,
}: {
  keys: readonly PromptTriggerKey[];
  loras: GenerateLora[];
  selectedModel: GenerateModelConfig | undefined;
  isDisabled?: boolean;
  onChange: (value: string) => void;
}): PromptTriggerAutocompleteApi => {
  const listboxId = useId();
  const optionIdPrefix = `${listboxId}-option-`;
  const [state, setState] = useState<AutocompleteState | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const isComposingRef = useRef(false);
  const options = usePromptTriggerOptions(loras, selectedModel);

  const matches = useMemo(
    () => (state ? getInlineTriggerOptions(options, state.query.key, state.query.query) : []),
    [options, state]
  );
  const isOpen = state !== null && matches.length > 0;
  const close = useCallback(() => setState(null), []);

  const refresh = useCallback(
    (textarea: HTMLTextAreaElement | null) => {
      if (!textarea || isDisabled || isComposingRef.current) {
        setState(null);
        return;
      }

      const query = getActiveTriggerQuery(textarea.value, textarea.selectionStart, keys);
      const caretRect = query ? getTextareaCaretRect(textarea, query.range.start) : null;

      setState(query && caretRect ? { caretRect, query, textarea } : null);
      setActiveIndex(0);
    },
    [isDisabled, keys]
  );

  const selectOption = useCallback(
    (option: PromptTriggerOption) => {
      if (state) {
        insertPromptText({
          onChange,
          range: state.query.range,
          text: option.value,
          textarea: state.textarea,
          value: state.textarea.value,
        });
      }

      setState(null);
    },
    [onChange, state]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>): void => {
      if (!isOpen || event.nativeEvent.isComposing || event.keyCode === 229) {
        return;
      }

      if (CARET_KEYS.includes(event.key)) {
        return;
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        setState(null);
        return;
      }

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        const step = event.key === 'ArrowDown' ? 1 : -1;

        event.preventDefault();
        setActiveIndex((current) => (current + step + matches.length) % matches.length);
        return;
      }

      if (event.key === 'Enter' || event.key === 'Tab') {
        const option = matches[activeIndex];

        if (option) {
          event.preventDefault();
          selectOption(option);
        }
      }
    },
    [activeIndex, isOpen, matches, selectOption]
  );

  const handleKeyUp = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (CARET_KEYS.includes(event.key) || (!isOpen && LIST_KEYS.includes(event.key))) {
        refresh(event.currentTarget);
      }
    },
    [isOpen, refresh]
  );

  const handleCompositionStart = useCallback(() => {
    isComposingRef.current = true;
    setState(null);
  }, []);

  const handleCompositionEnd = useCallback(
    (event: CompositionEvent<HTMLTextAreaElement>) => {
      isComposingRef.current = false;
      refresh(event.currentTarget);
    },
    [refresh]
  );

  return {
    close,
    comboboxProps: {
      'aria-activedescendant': isOpen ? `${optionIdPrefix}${activeIndex}` : undefined,
      'aria-autocomplete': 'list',
      'aria-controls': isOpen ? listboxId : undefined,
      'aria-expanded': isOpen,
      onCompositionEnd: handleCompositionEnd,
      onCompositionStart: handleCompositionStart,
      onKeyUp: handleKeyUp,
      role: 'combobox',
    },
    element:
      isOpen && state ? (
        <>
          <DismissOnViewportChange dismiss={close} enabled />
          <PromptTriggerAutocomplete
            activeIndex={activeIndex}
            caretRect={state.caretRect}
            listboxId={listboxId}
            optionIdPrefix={optionIdPrefix}
            options={matches}
            onSelect={selectOption}
          />
        </>
      ) : null,
    handleKeyDown,
    isOpen,
    refresh,
  };
};
