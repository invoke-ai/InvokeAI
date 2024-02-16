import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ShowDynamicPromptsPreviewButton } from 'features/dynamicPrompts/components/ShowDynamicPromptsPreviewButton';
import { AddEmbeddingButton } from 'features/embedding/AddEmbeddingButton';
import { EmbeddingPopover } from 'features/embedding/EmbeddingPopover';
import { usePrompt } from 'features/embedding/usePrompt';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { setPositivePrompt } from 'features/parameters/store/generationSlice';
import { SDXLConcatButton } from 'features/sdxl/components/SDXLPrompts/SDXLConcatButton';
import { memo, useCallback, useRef } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const ParamPositivePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector((s) => s.generation.positivePrompt);
  const baseModel = useAppSelector((s) => s.generation.model)?.base;

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(setPositivePrompt(v));
    },
    [dispatch]
  );
  const { onChange, isOpen, onClose, onOpen, onSelectEmbedding, onKeyDown, onFocus } = usePrompt({
    prompt,
    textareaRef: textareaRef,
    onChange: handleChange,
  });

  const focus: HotkeyCallback = useCallback(
    (e) => {
      onFocus();
      e.preventDefault();
    },
    [onFocus]
  );

  useHotkeys('alt+a', focus, []);

  return (
    <EmbeddingPopover
      isOpen={isOpen}
      onClose={onClose}
      onSelect={onSelectEmbedding}
      width={textareaRef.current?.clientWidth}
    >
      <Box pos="relative">
        <Textarea
          id="prompt"
          name="prompt"
          ref={textareaRef}
          value={prompt}
          placeholder={t('parameters.positivePromptPlaceholder')}
          onChange={onChange}
          minH={28}
          onKeyDown={onKeyDown}
          variant="darkFilled"
        />
        <PromptOverlayButtonWrapper>
          <AddEmbeddingButton isOpen={isOpen} onOpen={onOpen} />
          {baseModel === 'sdxl' && <SDXLConcatButton />}
          <ShowDynamicPromptsPreviewButton />
        </PromptOverlayButtonWrapper>
      </Box>
    </EmbeddingPopover>
  );
});

ParamPositivePrompt.displayName = 'ParamPositivePrompt';
