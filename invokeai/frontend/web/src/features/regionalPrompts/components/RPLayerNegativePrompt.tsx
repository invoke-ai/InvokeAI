import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { RPLayerPromptDeleteButton } from 'features/regionalPrompts/components/RPLayerPromptDeleteButton';
import { useLayerNegativePrompt } from 'features/regionalPrompts/hooks/layerStateHooks';
import { maskLayerNegativePromptChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  layerId: string;
};

export const RPLayerNegativePrompt = memo(({ layerId }: Props) => {
  const prompt = useLayerNegativePrompt(layerId);
  const dispatch = useAppDispatch();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const _onChange = useCallback(
    (v: string) => {
      dispatch(maskLayerNegativePromptChanged({ layerId, prompt: v }));
    },
    [dispatch, layerId]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown } = usePrompt({
    prompt,
    textareaRef,
    onChange: _onChange,
  });

  return (
    <PromptPopover isOpen={isOpen} onClose={onClose} onSelect={onSelect} width={textareaRef.current?.clientWidth}>
      <Box pos="relative" w="full">
        <Textarea
          id="prompt"
          name="prompt"
          ref={textareaRef}
          value={prompt}
          placeholder={t('parameters.negativePromptPlaceholder')}
          onChange={onChange}
          onKeyDown={onKeyDown}
          variant="darkFilled"
          paddingRight={30}
          fontSize="sm"
        />
        <PromptOverlayButtonWrapper>
          <RPLayerPromptDeleteButton layerId={layerId} polarity="negative" />
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
        </PromptOverlayButtonWrapper>
      </Box>
    </PromptPopover>
  );
});

RPLayerNegativePrompt.displayName = 'RPLayerNegativePrompt';
