import { Box, Flex, Textarea } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { usePersistedTextAreaSize } from 'common/hooks/usePersistedTextareaSize';
import {
  positivePromptAddedToHistory,
  positivePromptChanged,
  selectModelSupportsNegativePrompt,
  selectPositivePrompt,
  selectPositivePromptHistory,
} from 'features/controlLayers/store/paramsSlice';
import { promptGenerationFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { ShowDynamicPromptsPreviewButton } from 'features/dynamicPrompts/components/ShowDynamicPromptsPreviewButton';
import { NegativePromptToggleButton } from 'features/parameters/components/Core/NegativePromptToggleButton';
import { PromptLabel } from 'features/parameters/components/Prompts/PromptLabel';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { ViewModePrompt } from 'features/parameters/components/Prompts/ViewModePrompt';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptExpansionMenu } from 'features/prompt/PromptExpansion/PromptExpansionMenu';
import { PromptExpansionOverlay } from 'features/prompt/PromptExpansion/PromptExpansionOverlay';
import { promptExpansionApi } from 'features/prompt/PromptExpansion/state';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import {
  selectStylePresetActivePresetId,
  selectStylePresetViewMode,
} from 'features/stylePresets/store/stylePresetSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectAllowPromptExpansion } from 'features/system/store/configSlice';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import { PositivePromptHistoryIconButton } from './PositivePromptHistory';

const persistOptions: Parameters<typeof usePersistedTextAreaSize>[2] = {
  trackWidth: false,
  trackHeight: true,
  initialHeight: 120,
};

export const ParamPositivePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectPositivePrompt);
  const history = useAppSelector(selectPositivePromptHistory);
  const viewMode = useAppSelector(selectStylePresetViewMode);
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);
  const { isPending: isPromptExpansionPending } = useStore(promptExpansionApi.$state);
  const isPromptExpansionEnabled = useAppSelector(selectAllowPromptExpansion);
  const activeTab = useAppSelector(selectActiveTab);

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
    },
    [dispatch]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown, onFocus } = usePrompt({
    prompt,
    textareaRef: textareaRef,
    onChange: handleChange,
    isDisabled: isPromptExpansionPending,
  });

  // Browsing state for boundary Up/Down traversal
  const browsingIndexRef = useRef<number | null>(null); // null => not browsing; 0..n => index in history
  const preBrowsePromptRef = useRef<string>(''); // original prompt when browsing started
  const lastHistoryFirstRef = useRef<string | undefined>(undefined);

  // Reset browsing when history updates due to a new generation (first item changes or history mutates)
  useEffect(() => {
    if (lastHistoryFirstRef.current !== history[0]) {
      browsingIndexRef.current = null;
      preBrowsePromptRef.current = '';
      lastHistoryFirstRef.current = history[0];
    }
  }, [history]);

  // Boundary navigation via Up/Down keys was replaced by explicit hotkeys below.

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

  // Compute a starting working history and ensure current prompt is bumped into history
  const startBrowsing = useCallback(() => {
    if (browsingIndexRef.current !== null) {
      return;
    }
    preBrowsePromptRef.current = prompt ?? '';
    const trimmedCurrent = (prompt ?? '').trim();
    if (trimmedCurrent) {
      dispatch(positivePromptAddedToHistory(trimmedCurrent));
    }
    browsingIndexRef.current = 0;
  }, [dispatch, prompt]);

  const applyHistoryAtIndex = useCallback(
    (idx: number, placeCaretAt: 'start' | 'end') => {
      const list = history;
      if (list.length === 0) {
        return;
      }
      const clamped = Math.max(0, Math.min(idx, list.length - 1));
      browsingIndexRef.current = clamped;
      const historyItem = list[clamped];
      if (historyItem !== undefined) {
        dispatch(positivePromptChanged(historyItem));
      }
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (!el) {
          return;
        }
        if (placeCaretAt === 'start') {
          el.selectionStart = 0;
          el.selectionEnd = 0;
        } else {
          const end = el.value.length;
          el.selectionStart = end;
          el.selectionEnd = end;
        }
      });
    },
    [dispatch, history]
  );

  const browsePrev = useCallback(() => {
    if (!isPromptFocused()) {
      return;
    }
    if (history.length === 0) {
      return;
    }
    if (browsingIndexRef.current === null) {
      startBrowsing();
      // Move to older entry on first activation
      if (history.length > 1) {
        applyHistoryAtIndex(1, 'start');
      } else {
        applyHistoryAtIndex(0, 'start');
      }
      return;
    }
    // Already browsing, go older if possible
    const next = Math.min((browsingIndexRef.current ?? 0) + 1, history.length - 1);
    applyHistoryAtIndex(next, 'start');
  }, [applyHistoryAtIndex, history.length, isPromptFocused, startBrowsing]);

  const browseNext = useCallback(() => {
    if (!isPromptFocused()) {
      return;
    }
    if (history.length === 0) {
      return;
    }
    if (browsingIndexRef.current === null) {
      // Not browsing; Down does nothing (matches shell semantics)
      return;
    }
    if ((browsingIndexRef.current ?? 0) > 0) {
      const next = (browsingIndexRef.current ?? 0) - 1;
      applyHistoryAtIndex(next, 'end');
    } else {
      // Exit browsing and restore pre-browse prompt
      browsingIndexRef.current = null;
      dispatch(positivePromptChanged(preBrowsePromptRef.current));
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (el) {
          const end = el.value.length;
          el.selectionStart = end;
          el.selectionEnd = end;
        }
      });
    }
  }, [applyHistoryAtIndex, dispatch, history.length, isPromptFocused]);

  // Register hotkeys for browsing
  useRegisteredHotkeys({
    id: 'promptHistoryPrev',
    category: 'app',
    callback: (e) => {
      if (isPromptFocused()) {
        e.preventDefault();
        browsePrev();
      }
    },
    options: { preventDefault: true, enableOnFormTags: ['INPUT', 'SELECT', 'TEXTAREA'] },
    dependencies: [browsePrev, isPromptFocused],
  });
  useRegisteredHotkeys({
    id: 'promptHistoryNext',
    category: 'app',
    callback: (e) => {
      if (isPromptFocused()) {
        e.preventDefault();
        browseNext();
      }
    },
    options: { preventDefault: true, enableOnFormTags: ['INPUT', 'SELECT', 'TEXTAREA'] },
    dependencies: [browseNext, isPromptFocused],
  });

  const dndTargetData = useMemo(() => promptGenerationFromImageDndTarget.getData(), []);

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
            minH={isPromptExpansionEnabled ? 44 : 32}
            isDisabled={isPromptExpansionPending}
          />
          <PromptOverlayButtonWrapper>
            <Flex flexDir="column" gap={2} justifyContent="flex-start" alignItems="center">
              <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
              <ShowDynamicPromptsPreviewButton />
              <PositivePromptHistoryIconButton />
              {activeTab !== 'video' && modelSupportsNegativePrompt && <NegativePromptToggleButton />}
            </Flex>
            {isPromptExpansionEnabled && <PromptExpansionMenu />}
          </PromptOverlayButtonWrapper>
          <PromptLabel label="Prompt" />
          {viewMode && (
            <ViewModePrompt
              prompt={prompt}
              presetPrompt={activeStylePreset?.preset_data.positive_prompt || ''}
              label={`${t('parameters.positivePromptPlaceholder')} (${t('stylePresets.preview')})`}
            />
          )}
          {isPromptExpansionEnabled && (
            <DndDropTarget
              dndTarget={promptGenerationFromImageDndTarget}
              dndTargetData={dndTargetData}
              label={t('prompt.generateFromImage')}
              isDisabled={isPromptExpansionPending}
            />
          )}
          <PromptExpansionOverlay />
        </Box>
      </PromptPopover>
    </Box>
  );
});

ParamPositivePrompt.displayName = 'ParamPositivePrompt';
