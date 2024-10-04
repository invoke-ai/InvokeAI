import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { positivePromptChanged, selectBase, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { ShowDynamicPromptsPreviewButton } from 'features/dynamicPrompts/components/ShowDynamicPromptsPreviewButton';
import { PromptLabel } from 'features/parameters/components/Prompts/PromptLabel';
import { PromptOverlayButtonWrapper } from 'features/parameters/components/Prompts/PromptOverlayButtonWrapper';
import { ViewModePrompt } from 'features/parameters/components/Prompts/ViewModePrompt';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { SDXLConcatButton } from 'features/sdxl/components/SDXLPrompts/SDXLConcatButton';
import {
  selectStylePresetActivePresetId,
  selectStylePresetViewMode,
} from 'features/stylePresets/store/stylePresetSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback, useRef } from 'react';
import type { HotkeyCallback } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

export const ParamPositivePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectPositivePrompt);
  const baseModel = useAppSelector(selectBase);
  const viewMode = useAppSelector(selectStylePresetViewMode);
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);

  const { activeStylePreset } = useListStylePresetsQuery(undefined, {
    selectFromResult: ({ data }) => {
      let activeStylePreset = null;
      if (data) {
        activeStylePreset = data.find((sp) => sp.id === activeStylePresetId);
      }
      return { activeStylePreset };
    },
  });

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: string) => {
      dispatch(positivePromptChanged(v));
    },
    [dispatch]
  );
  const { onChange, isOpen, onClose, onOpen, onSelect, onKeyDown, onFocus } = usePrompt({
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

  useRegisteredHotkeys({
    id: 'focusPrompt',
    category: 'app',
    callback: focus,
    options: { preventDefault: true, enableOnFormTags: ['INPUT', 'SELECT', 'TEXTAREA'] },
    dependencies: [focus],
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
          minH={40}
          onKeyDown={onKeyDown}
          variant="darkFilled"
          borderTopWidth={24} // This prevents the prompt from being hidden behind the header
          paddingInlineEnd={10}
          paddingInlineStart={3}
          paddingTop={0}
          paddingBottom={3}
        />
        <PromptOverlayButtonWrapper>
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
          {baseModel === 'sdxl' && <SDXLConcatButton />}
          <ShowDynamicPromptsPreviewButton />
        </PromptOverlayButtonWrapper>
        <PromptLabel label={t('parameters.positivePromptPlaceholder')} />
        {viewMode && (
          <ViewModePrompt
            prompt={prompt}
            presetPrompt={activeStylePreset?.preset_data.positive_prompt || ''}
            label={`${t('parameters.positivePromptPlaceholder')} (${t('stylePresets.preview')})`}
          />
        )}
      </Box>
    </PromptPopover>
  );
});

ParamPositivePrompt.displayName = 'ParamPositivePrompt';
