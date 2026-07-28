/**
 * Type-as-you-filter trigger insertion, anchored at the caret.
 *
 * The old behaviour swallowed the `__` or `<` keystroke, opened a popover
 * anchored to the whole textarea, and gave you a separate search box: in a tall
 * prompt box the list appeared at the bottom-left corner, and you typed in one
 * place while looking at another. Here the trigger is typed like any other
 * character, whatever follows it narrows the list, and the list sits beside the
 * caret. Nothing is swallowed, so there is nothing to give back on dismissal.
 *
 * Focus never leaves the textarea. That is what makes this a hook returning an
 * element rather than a `Popover`: the list is a passive surface, and every key
 * that drives it is handled on the textarea itself.
 */

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
import { useCallback, useId, useMemo, useRef, useState } from 'react';

interface AutocompleteState {
  caretRect: CaretRect;
  query: PromptTriggerQuery;
  /**
   * The field the query was read from. Held rather than looked up again at
   * selection time: `document.activeElement` is only the textarea because the
   * option suppressed its own mousedown, and if that ever stops holding, the
   * insert reads `undefined` for the current value and commits the option on its
   * own — silently replacing the whole prompt.
   */
  textarea: HTMLTextAreaElement;
}

export interface PromptTriggerAutocompleteApi {
  /**
   * Spread onto the textarea so assistive tech sees a combobox, and so the list
   * stays out of the way while an IME is composing.
   */
  comboboxProps: {
    'aria-activedescendant': string | undefined;
    'aria-autocomplete': 'list';
    'aria-controls': string | undefined;
    'aria-expanded': boolean;
    'aria-multiline': true;
    role: 'combobox';
    onCompositionEnd: (event: CompositionEvent<HTMLTextAreaElement>) => void;
    onCompositionStart: () => void;
  };
  element: ReactNode;
  isOpen: boolean;
  /** Close without inserting, for anything that invalidates the caret. */
  close: () => void;
  /** Re-read the trigger under the caret. Call on input, and on click. */
  refresh: (textarea: HTMLTextAreaElement | null) => void;
  /** `true` when the key drove the list and the field should not also act on it. */
  handleKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => boolean;
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
      // Mid-composition the field holds half-committed romaji, which is not a
      // wildcard name being typed. Reading it would re-anchor the list on every
      // keystroke the IME is still working on.
      if (!textarea || isDisabled || isComposingRef.current) {
        setState(null);
        return;
      }

      const query = getActiveTriggerQuery(textarea.value, textarea.selectionStart, keys);
      // Anchored at the trigger rather than the caret, so the list holds still
      // while the name is typed instead of creeping sideways with it.
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
    (event: KeyboardEvent<HTMLTextAreaElement>): boolean => {
      // While an IME is composing, the arrows move through its candidate window
      // and Enter commits the composition. Taking either would leave a Japanese
      // or Chinese user unable to pick a candidate at all. `keyCode === 229` is
      // the same signal for the browsers that do not set `isComposing`.
      if (!isOpen || event.nativeEvent.isComposing || event.keyCode === 229) {
        return false;
      }

      // A caret jump leaves the query behind, so the list goes with it — but the
      // key still belongs to the textarea.
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        setState(null);
        return false;
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        // Otherwise the popover or dialog this prompt sits in closes too.
        event.stopPropagation();
        setState(null);
        return true;
      }

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        const step = event.key === 'ArrowDown' ? 1 : -1;

        event.preventDefault();
        setActiveIndex((current) => (current + step + matches.length) % matches.length);
        return true;
      }

      if (event.key === 'Enter' || event.key === 'Tab') {
        const option = matches[activeIndex];

        if (option) {
          event.preventDefault();
          selectOption(option);
          return true;
        }
      }

      return false;
    },
    [activeIndex, isOpen, matches, selectOption]
  );

  const handleCompositionStart = useCallback(() => {
    isComposingRef.current = true;
    setState(null);
  }, []);

  const handleCompositionEnd = useCallback(
    (event: CompositionEvent<HTMLTextAreaElement>) => {
      isComposingRef.current = false;
      // The composed text is only in the field now, so this is the first point
      // at which a trigger under the caret is worth reading again.
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
      // `combobox` overrides the textarea's implicit `textbox` role, which is
      // what carries multiline-ness; without this a tall prompt box announces
      // as a single-line field.
      'aria-multiline': true,
      onCompositionEnd: handleCompositionEnd,
      onCompositionStart: handleCompositionStart,
      role: 'combobox',
    },
    element:
      isOpen && state ? (
        <PromptTriggerAutocomplete
          activeIndex={activeIndex}
          caretRect={state.caretRect}
          listboxId={listboxId}
          optionIdPrefix={optionIdPrefix}
          options={matches}
          onSelect={selectOption}
        />
      ) : null,
    handleKeyDown,
    isOpen,
    refresh,
  };
};
