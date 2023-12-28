import { Box } from '@chakra-ui/layout';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvAutosizeTextarea } from 'common/components/InvAutosizeTextarea/InvAutosizeTextarea';
import { AddEmbeddingButton } from 'features/embedding/AddEmbeddingButton';
import { EmbeddingPopover } from 'features/embedding/EmbeddingPopover';
import { usePrompt } from 'features/embedding/usePrompt';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { setNegativeStylePromptSDXL } from 'features/sdxl/store/sdxlSlice';
import { useCallback, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const ParamSDXLNegativeStylePrompt = () => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector((state) => state.sdxl.negativeStylePrompt);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(setNegativeStylePromptSDXL(v));
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

  useHotkeys('alt+a', onFocus, []);

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
          placeholder={t('sdxl.negStylePrompt')}
          onChange={onChange}
          onKeyDown={onKeyDown}
          minH="unset"
          fontSize="sm"
          minRows={2}
          maxRows={5}
          variant="darkFilled"
        />
        <PromptOverlayButtonWrapper>
          <AddEmbeddingButton isOpen={isOpen} onOpen={onOpen} />
        </PromptOverlayButtonWrapper>
      </Box>
    </EmbeddingPopover>
  );
};
