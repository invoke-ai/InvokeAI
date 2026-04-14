import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForElements, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Box, Flex, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { usePersistedTextAreaSize } from 'common/hooks/usePersistedTextareaSize';
import {
  positivePromptChanged,
  selectModelSupportsNegativePrompt,
  selectPositivePrompt,
  selectPositivePromptHistory,
} from 'features/controlLayers/store/paramsSlice';
import { singleImageDndSource } from 'features/dnd/dnd';
import { DndDropOverlay } from 'features/dnd/DndDropOverlay';
import type { DndTargetState } from 'features/dnd/types';
import { ShowDynamicPromptsPreviewButton } from 'features/dynamicPrompts/components/ShowDynamicPromptsPreviewButton';
import { NegativePromptToggleButton } from 'features/parameters/components/Core/NegativePromptToggleButton';
import { PromptLabel } from 'features/parameters/components/Prompts/PromptLabel';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { PromptResizeHandle } from 'features/parameters/components/Prompts/PromptResizeHandle';
import { ViewModePrompt } from 'features/parameters/components/Prompts/ViewModePrompt';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { ExpandPromptButton } from 'features/prompt/ExpandPromptButton';
import { ImageToPromptButton } from 'features/prompt/ImageToPromptButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { clearPromptUndo, consumePromptUndo } from 'features/prompt/promptUndo';
import { usePrompt } from 'features/prompt/usePrompt';
import { usePromptAttentionHotkeys } from 'features/prompt/usePromptAttentionHotkeys';
import {
  selectStylePresetActivePresetId,
  selectStylePresetViewMode,
} from 'features/stylePresets/store/stylePresetSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useClickAway } from 'react-use';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';
import { useLlavaModels } from 'services/api/hooks/modelsByType';
import type { ImageDTO } from 'services/api/types';

import { PositivePromptHistoryIconButton } from './PositivePromptHistory';

const persistOptions: Parameters<typeof usePersistedTextAreaSize>[2] = {
  trackWidth: false,
  trackHeight: true,
  initialHeight: 120,
};

const POSITIVE_PROMPT_MIN_HEIGHT = 32;

const usePromptHistory = () => {
  const store = useAppStore();
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
    store.dispatch(positivePromptChanged(newPrompt));
  }, [history, store]);
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
      store.dispatch(positivePromptChanged(state.stashedPrompt));
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
    store.dispatch(positivePromptChanged(newPrompt));
  }, [history, store]);
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
  const [llavaModels] = useLlavaModels();
  const hasLlavaModels = llavaModels.length > 0;

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
      // Clear LLM undo state when the user types manually
      clearPromptUndo();
    },
    [dispatch, promptHistoryApi]
  );
  const {
    onChange,
    isOpen,
    onClose,
    onOpen,
    onSelect,
    onKeyDown: onKeyDownPrompt,
    onFocus,
  } = usePrompt({
    prompt,
    textareaRef: textareaRef,
    onChange: handleChange,
  });

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Intercept Ctrl+Z to undo LLM prompt changes
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        const previousPrompt = consumePromptUndo();
        if (previousPrompt !== null) {
          e.preventDefault();
          dispatch(positivePromptChanged(previousPrompt));
          return;
        }
      }
      onKeyDownPrompt(e);
    },
    [dispatch, onKeyDownPrompt]
  );

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

  usePromptAttentionHotkeys({
    textareaRef,
    onPromptChange: (prompt) => dispatch(positivePromptChanged(prompt)),
  });

  // Drop target for gallery images -> Image to Prompt
  const dropTargetRef = useRef<HTMLDivElement>(null);
  const [droppedImage, setDroppedImage] = useState<ImageDTO | undefined>(undefined);
  const [dndState, setDndState] = useState<DndTargetState>('idle');

  const clearDroppedImage = useCallback(() => {
    setDroppedImage(undefined);
  }, []);

  useEffect(() => {
    const element = dropTargetRef.current;
    if (!element || !hasLlavaModels) {
      return;
    }

    return combine(
      dropTargetForElements({
        element,
        canDrop: ({ source }) => singleImageDndSource.typeGuard(source.data),
        onDragEnter: () => setDndState('over'),
        onDragLeave: () => setDndState('potential'),
        onDrop: ({ source }) => {
          setDndState('idle');
          if (singleImageDndSource.typeGuard(source.data)) {
            setDroppedImage(source.data.payload.imageDTO);
          }
        },
      }),
      monitorForElements({
        canMonitor: ({ source }) => singleImageDndSource.typeGuard(source.data),
        onDragStart: () => setDndState('potential'),
        onDrop: () => setDndState('idle'),
      })
    );
  }, [hasLlavaModels]);

  return (
    <Box pos="relative" ref={dropTargetRef}>
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
            resize="none"
            minH={POSITIVE_PROMPT_MIN_HEIGHT}
            fontFamily="mono"
            fontSize="0.82rem"
            sx={{ '&::-webkit-resizer': { display: 'none' } }}
          />
          <PromptOverlayButtonWrapper>
            <Flex flexDir="column" gap={2} justifyContent="flex-start" alignItems="center">
              <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
              <ShowDynamicPromptsPreviewButton />
              <ExpandPromptButton />
              <ImageToPromptButton droppedImage={droppedImage} onClearDroppedImage={clearDroppedImage} />
              <PositivePromptHistoryIconButton />
              {modelSupportsNegativePrompt && <NegativePromptToggleButton />}
            </Flex>
          </PromptOverlayButtonWrapper>
          <PromptLabel label={t('controlLayers.prompt')} />
          {viewMode && (
            <ViewModePrompt
              prompt={prompt}
              presetPrompt={activeStylePreset?.preset_data.positive_prompt || ''}
              label={`${t('parameters.positivePromptPlaceholder')} (${t('stylePresets.preview')})`}
            />
          )}
          <PromptResizeHandle textareaRef={textareaRef} minHeight={POSITIVE_PROMPT_MIN_HEIGHT} />
        </Box>
      </PromptPopover>
      {hasLlavaModels && <DndDropOverlay dndState={dndState} label={t('prompt.imageToPrompt')} />}
    </Box>
  );
});

ParamPositivePrompt.displayName = 'ParamPositivePrompt';
