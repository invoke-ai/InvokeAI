import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { negativePromptChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamNegativePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector((s) => s.regionalPrompts.present.baseLayer.negativePrompt);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const _onChange = useCallback(
    (v: string) => {
      dispatch(negativePromptChanged(v));
    },
    [dispatch]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown } = usePrompt({
    prompt,
    textareaRef,
    onChange: _onChange,
  });

  return (
    <PromptPopover isOpen={isOpen} onClose={onClose} onSelect={onSelect} width={textareaRef.current?.clientWidth}>
      <Box pos="relative">
        <Textarea
          id="negativePrompt"
          name="negativePrompt"
          ref={textareaRef}
          value={prompt}
          placeholder={t('parameters.negativePromptPlaceholder')}
          onChange={onChange}
          onKeyDown={onKeyDown}
          fontSize="sm"
          variant="darkFilled"
          paddingRight={30}
        />
        <PromptOverlayButtonWrapper>
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
        </PromptOverlayButtonWrapper>
      </Box>
    </PromptPopover>
  );
});

ParamNegativePrompt.displayName = 'ParamNegativePrompt';
