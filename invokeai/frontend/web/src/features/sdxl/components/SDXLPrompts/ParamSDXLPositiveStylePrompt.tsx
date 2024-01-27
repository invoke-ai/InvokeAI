import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { AddEmbeddingButton } from 'features/embedding/AddEmbeddingButton';
import { EmbeddingPopover } from 'features/embedding/EmbeddingPopover';
import { usePrompt } from 'features/embedding/usePrompt';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { setPositiveStylePromptSDXL } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamSDXLPositiveStylePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector((s) => s.sdxl.positiveStylePrompt);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(setPositiveStylePromptSDXL(v));
    },
    [dispatch]
  );
  const { onChange, isOpen, onClose, onOpen, onSelectEmbedding, onKeyDown } = usePrompt({
    prompt,
    textareaRef: textareaRef,
    onChange: handleChange,
  });

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
          placeholder={t('sdxl.posStylePrompt')}
          onChange={onChange}
          onKeyDown={onKeyDown}
          fontSize="sm"
          variant="darkFilled"
        />
        <PromptOverlayButtonWrapper>
          <AddEmbeddingButton isOpen={isOpen} onOpen={onOpen} />
        </PromptOverlayButtonWrapper>
      </Box>
    </EmbeddingPopover>
  );
});

ParamSDXLPositiveStylePrompt.displayName = 'ParamSDXLPositiveStylePrompt';
