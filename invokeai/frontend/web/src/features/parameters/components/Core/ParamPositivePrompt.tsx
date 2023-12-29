import { Box } from '@chakra-ui/layout';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvAutosizeTextarea } from 'common/components/InvAutosizeTextarea/InvAutosizeTextarea';
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
  const prompt = useAppSelector((state) => state.generation.positivePrompt);
  const baseModel = useAppSelector((state) => state.generation.model)
    ?.base_model;

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(setPositivePrompt(v));
    },
    [dispatch]
  );
  const {
    onChange,
    isOpen,
    onClose,
    onOpen,
    onSelectEmbedding,
    onKeyDown,
    onFocus,
  } = usePrompt({
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
        <InvAutosizeTextarea
          id="prompt"
          name="prompt"
          ref={textareaRef}
          value={prompt}
          placeholder={t('parameters.positivePromptPlaceholder')}
          onChange={onChange}
          onKeyDown={onKeyDown}
          minRows={4}
          maxRows={7}
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
