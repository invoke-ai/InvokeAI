import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { usePersistedTextAreaSize } from 'common/hooks/usePersistedTextareaSize';
import { negativePrompt2Changed, selectNegativePrompt2 } from 'features/controlLayers/store/paramsSlice';
import { PromptLabel } from 'features/parameters/components/Prompts/PromptLabel';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const persistOptions: Parameters<typeof usePersistedTextAreaSize>[2] = {
  trackWidth: false,
  trackHeight: true,
};

export const ParamSDXLNegativeStylePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectNegativePrompt2);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  usePersistedTextAreaSize('negative_style_prompt', textareaRef, persistOptions);

  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(negativePrompt2Changed(v));
    },
    [dispatch]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown } = usePrompt({
    prompt,
    textareaRef: textareaRef,
    onChange: handleChange,
  });

  return (
    <PromptPopover isOpen={isOpen} onClose={onClose} onSelect={onSelect} width={textareaRef.current?.clientWidth}>
      <Box pos="relative">
        <Textarea
          id="prompt"
          name="prompt"
          ref={textareaRef}
          value={prompt}
          onChange={onChange}
          onKeyDown={onKeyDown}
          fontSize="sm"
          variant="darkFilled"
          minH={24}
          borderTopWidth={24} // This prevents the prompt from being hidden behind the header
          paddingInlineEnd={10}
          paddingInlineStart={3}
          paddingTop={0}
          paddingBottom={3}
        />
        <PromptOverlayButtonWrapper>
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
        </PromptOverlayButtonWrapper>
        <PromptLabel label={t('sdxl.negStylePrompt')} />
      </Box>
    </PromptPopover>
  );
});

ParamSDXLNegativeStylePrompt.displayName = 'ParamSDXLNegativeStylePrompt';
