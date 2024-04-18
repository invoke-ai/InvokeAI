import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { useLayerPositivePrompt } from 'features/regionalPrompts/hooks/layerStateHooks';
import { positivePromptChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useRef } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

type Props = {
  layerId: string;
};

export const RegionalPromptsPositivePrompt = memo((props: Props) => {
  const prompt = useLayerPositivePrompt(props.layerId);
  const dispatch = useAppDispatch();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const _onChange = useCallback(
    (v: string) => {
      dispatch(positivePromptChanged({ layerId: props.layerId, prompt: v }));
    },
    [dispatch, props.layerId]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown, onFocus } = usePrompt({
    prompt,
    textareaRef,
    onChange: _onChange,
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
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
        </PromptOverlayButtonWrapper>
      </Box>
    </PromptPopover>
  );
});

RegionalPromptsPositivePrompt.displayName = 'RegionalPromptsPrompt';
