import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { positivePromptChanged } from 'features/controlLayers/store/controlLayersSlice';
import { ShowDynamicPromptsPreviewButton } from 'features/dynamicPrompts/components/ShowDynamicPromptsPreviewButton';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { ViewModePrompt } from 'features/parameters/components/Prompts/ViewModePrompt';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { SDXLConcatButton } from 'features/sdxl/components/SDXLPrompts/SDXLConcatButton';
import { memo, useCallback, useRef } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

const DEFAULT_HEIGHT = 28;

export const ParamPositivePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector((s) => s.controlLayers.present.positivePrompt);
  const baseModel = useAppSelector((s) => s.generation.model)?.base;
  const viewMode = useAppSelector((s) => s.stylePreset.viewMode);
  const activeStylePreset = useAppSelector((s) => s.stylePreset.activeStylePreset);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(positivePromptChanged(v));
    },
    [dispatch]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown, onFocus, onFocusCursorAtEnd } = usePrompt({
    prompt,
    textareaRef: textareaRef,
    onChange: handleChange,
  });

  const handleFocus = useCallback(() => {
    setTimeout(() => {
      onFocusCursorAtEnd();
    }, 500);
  }, [onFocusCursorAtEnd]);

  const focus: HotkeyCallback = useCallback(
    (e) => {
      onFocus();
      e.preventDefault();
    },
    [onFocus]
  );

  useHotkeys('alt+a', focus, []);

  if (viewMode) {
    return (
      <ViewModePrompt
        prompt={prompt}
        presetPrompt={activeStylePreset?.preset_data.positive_prompt || ''}
        height={DEFAULT_HEIGHT}
        onExit={handleFocus}
      />
    );
  }

  return (
    <PromptPopover isOpen={isOpen} onClose={onClose} onSelect={onSelect} width={textareaRef.current?.clientWidth}>
      <Box pos="relative">
        <Textarea
          id="prompt"
          name="prompt"
          ref={textareaRef}
          value={prompt}
          placeholder={t('parameters.globalPositivePromptPlaceholder')}
          onChange={onChange}
          minH={DEFAULT_HEIGHT}
          onKeyDown={onKeyDown}
          variant="darkFilled"
          paddingRight={30}
        />
        <PromptOverlayButtonWrapper>
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
          {baseModel === 'sdxl' && <SDXLConcatButton />}
          <ShowDynamicPromptsPreviewButton />
        </PromptOverlayButtonWrapper>
      </Box>
    </PromptPopover>
  );
});

ParamPositivePrompt.displayName = 'ParamPositivePrompt';
