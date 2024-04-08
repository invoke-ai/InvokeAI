import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ShowDynamicPromptsPreviewButton } from 'features/dynamicPrompts/components/ShowDynamicPromptsPreviewButton';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { useLayer } from 'features/regionalPrompts/hooks/useLayer';
import { promptChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { SDXLConcatButton } from 'features/sdxl/components/SDXLPrompts/SDXLConcatButton';
import { memo, useCallback, useRef } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

type Props = {
  layerId: string;
};

export const RegionalPromptsPrompt = memo((props: Props) => {
  const layer = useLayer(props.layerId);
  const dispatch = useAppDispatch();
  const baseModel = useAppSelector((s) => s.generation.model)?.base;
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(promptChanged({ layerId: props.layerId, prompt: v }));
    },
    [dispatch, props.layerId]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown, onFocus } = usePrompt({
    prompt: layer.prompt,
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
    <PromptPopover isOpen={isOpen} onClose={onClose} onSelect={onSelect} width={textareaRef.current?.clientWidth}>
      <Box pos="relative">
        <Textarea
          id="prompt"
          name="prompt"
          ref={textareaRef}
          value={layer.prompt}
          placeholder={t('parameters.positivePromptPlaceholder')}
          onChange={onChange}
          minH={28}
          minW={64}
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

RegionalPromptsPrompt.displayName = 'RegionalPromptsPrompt';
