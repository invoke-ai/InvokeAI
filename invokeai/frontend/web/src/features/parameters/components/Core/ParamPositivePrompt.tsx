import { Box, Flex, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { usePersistedTextAreaSize } from 'common/hooks/usePersistedTextareaSize';
import {
  positivePromptChanged,
  selectModelSupportsNegativePrompt,
  selectPositivePrompt,
  selectPositivePromptHistory,
} from 'features/controlLayers/store/paramsSlice';
import { ShowDynamicPromptsPreviewButton } from 'features/dynamicPrompts/components/ShowDynamicPromptsPreviewButton';
import { NegativePromptToggleButton } from 'features/parameters/components/Core/NegativePromptToggleButton';
import { PromptLabel } from 'features/parameters/components/Prompts/PromptLabel';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { ViewModePrompt } from 'features/parameters/components/Prompts/ViewModePrompt';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import {
  selectStylePresetActivePresetId,
  selectStylePresetViewMode,
} from 'features/stylePresets/store/stylePresetSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import React, { memo, useCallback, useRef } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useClickAway } from 'react-use';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import { PositivePromptHistoryIconButton } from './PositivePromptHistory';

const persistOptions: Parameters<typeof usePersistedTextAreaSize>[2] = {
  trackWidth: false,
  trackHeight: true,
  initialHeight: 120,
};

const usePromptHistory = () => {
  const store = useAppStore();
  const dispatch = useAppDispatch();
  const history = useAppSelector(selectPositivePromptHistory);

  /**
   * This ref is populated only when the user navigates back in history. In other words, its presence is a proxy
   * for "are we currently browsing history?"
   *
   * When we are moving thru history, we will always have a stashedPrompt (the prompt before we started browsing)
   * and a historyIdx which is an index into the history array (0 = most recent, 1 = previous, etc).
   */
  const stateRef = useRef<{ stashedPrompt: string; historyIdx: number } | null>(null);

  const prev = useCallback(() => {
    if (history.length === 0) {
      // No history, nothing to do
      return;
    }
    let state = stateRef.current;
    if (!state) {
      // First time going "back" in history, init state
      state = { stashedPrompt: selectPositivePrompt(store.getState()), historyIdx: 0 };
      stateRef.current = state;
    } else {
      // Subsequent "back" in history, increment index
      if (state.historyIdx === history.length - 1) {
        // Already at the end of history, nothing to do
        return;
      }
      state.historyIdx = state.historyIdx + 1;
    }
    // We should go "back" in history
    const newPrompt = history[state.historyIdx];
    if (newPrompt === undefined) {
      // Shouldn't happen
      return;
    }
    dispatch(positivePromptChanged(newPrompt));
  }, [dispatch, history, store]);
  const next = useCallback(() => {
    if (history.length === 0) {
      // No history, nothing to do
      return;
    }
    let state = stateRef.current;
    if (!state) {
      // If the user hasn't gone "back" in history, "forward" does nothing
      return;
    }
    state.historyIdx = state.historyIdx - 1;
    if (state.historyIdx < 0) {
      // Overshot to the "current" stashed prompt
      dispatch(positivePromptChanged(state.stashedPrompt));
      // Clear state bc we're back to current prompt
      stateRef.current = null;
      return;
    }
    // We should go "forward" in history
    const newPrompt = history[state.historyIdx];
    if (newPrompt === undefined) {
      // Shouldn't happen
      return;
    }
    dispatch(positivePromptChanged(newPrompt));
  }, [dispatch, history]);
  const reset = useCallback(() => {
    // Clear stashed state - used when user clicks away or types in the prompt box
    stateRef.current = null;
  }, []);
  return { prev, next, reset };
};

export const ParamPositivePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectPositivePrompt);
  const viewMode = useAppSelector(selectStylePresetViewMode);
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);

  const promptHistoryApi = usePromptHistory();

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  usePersistedTextAreaSize('positive_prompt', textareaRef, persistOptions);

  const { activeStylePreset } = useListStylePresetsQuery(undefined, {
    selectFromResult: ({ data }) => {
      let activeStylePreset = null;
      if (data) {
        activeStylePreset = data.find((sp) => sp.id === activeStylePresetId);
      }
      return { activeStylePreset };
    },
  });

  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(positivePromptChanged(v));
      // When the user changes the prompt, reset the prompt history state. This event is not fired when the prompt is
      // changed via the prompt history navigation.
      promptHistoryApi.reset();
    },
    [dispatch, promptHistoryApi]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown, onFocus } = usePrompt({
    prompt,
    textareaRef: textareaRef,
    onChange: handleChange,
  });

  // When the user clicks away from the textarea, reset the prompt history state.
  useClickAway(textareaRef, promptHistoryApi.reset);

  const focus: HotkeyCallback = useCallback(
    (e) => {
      onFocus();
      e.preventDefault();
    },
    [onFocus]
  );

  useRegisteredHotkeys({
    id: 'focusPrompt',
    category: 'app',
    callback: focus,
    options: { preventDefault: true, enableOnFormTags: ['INPUT', 'SELECT', 'TEXTAREA'] },
    dependencies: [focus],
  });

  // Helper: check if prompt textarea is focused
  const isPromptFocused = useCallback(() => document.activeElement === textareaRef.current, []);

  // Register hotkeys for browsing
  useRegisteredHotkeys({
    id: 'promptHistoryPrev',
    category: 'app',
    callback: (e) => {
      if (isPromptFocused()) {
        e.preventDefault();
        promptHistoryApi.prev();
      }
    },
    options: { preventDefault: true, enableOnFormTags: ['INPUT', 'SELECT', 'TEXTAREA'] },
    dependencies: [promptHistoryApi.prev, isPromptFocused],
  });
  useRegisteredHotkeys({
    id: 'promptHistoryNext',
    category: 'app',
    callback: (e) => {
      if (isPromptFocused()) {
        e.preventDefault();
        promptHistoryApi.next();
      }
    },
    options: { preventDefault: true, enableOnFormTags: ['INPUT', 'SELECT', 'TEXTAREA'] },
    dependencies: [promptHistoryApi.next, isPromptFocused],
  });

  return (
    <Box pos="relative">
      <PromptPopover isOpen={isOpen} onClose={onClose} onSelect={onSelect} width={textareaRef.current?.clientWidth}>
        <Box pos="relative">
          <Textarea
            className="positive-prompt-textarea"
            name="prompt"
            ref={textareaRef}
            value={prompt}
            onChange={onChange}
            onKeyDown={onKeyDown}
            variant="darkFilled"
            borderTopWidth={24} // This prevents the prompt from being hidden behind the header
            paddingInlineEnd={10}
            paddingInlineStart={3}
            paddingTop={0}
            paddingBottom={3}
            resize="vertical"
            minH={32}
          />
          <PromptOverlayButtonWrapper>
            <Flex flexDir="column" gap={2} justifyContent="flex-start" alignItems="center">
              <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
              <ShowDynamicPromptsPreviewButton />
              <PositivePromptHistoryIconButton />
              {modelSupportsNegativePrompt && <NegativePromptToggleButton />}
            </Flex>
          </PromptOverlayButtonWrapper>
          <PromptLabel label="Prompt" />
          {viewMode && (
            <ViewModePrompt
              prompt={prompt}
              presetPrompt={activeStylePreset?.preset_data.positive_prompt || ''}
              label={`${t('parameters.positivePromptPlaceholder')} (${t('stylePresets.preview')})`}
            />
          )}
        </Box>
      </PromptPopover>
    </Box>
  );
});

ParamPositivePrompt.displayName = 'ParamPositivePrompt';
