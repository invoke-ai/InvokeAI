import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RGDeletePromptButton } from 'features/controlLayers/components/RegionalGuidance/RGDeletePromptButton';
import { rgPositivePromptChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectRGOrThrow } from 'features/controlLayers/store/regionsReducers';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const RGPositivePrompt = memo(({ id }: Props) => {
  const prompt = useAppSelector((s) => selectRGOrThrow(s.canvasV2, id).positivePrompt ?? '');
  const dispatch = useAppDispatch();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const _onChange = useCallback(
    (v: string) => {
      dispatch(rgPositivePromptChanged({ id, prompt: v }));
    },
    [dispatch, id]
  );
  const onDeletePrompt = useCallback(() => {
    dispatch(rgPositivePromptChanged({ id, prompt: null }));
  }, [dispatch, id]);
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
          placeholder={t('parameters.positivePromptPlaceholder')}
          onChange={onChange}
          onKeyDown={onKeyDown}
          variant="darkFilled"
          paddingRight={30}
          minH={28}
        />
        <PromptOverlayButtonWrapper>
          <RGDeletePromptButton onDelete={onDeletePrompt} />
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
        </PromptOverlayButtonWrapper>
      </Box>
    </PromptPopover>
  );
});

RGPositivePrompt.displayName = 'RGPositivePrompt';
