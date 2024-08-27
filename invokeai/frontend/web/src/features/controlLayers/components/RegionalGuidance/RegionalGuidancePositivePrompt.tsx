import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Textarea } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RegionalGuidanceDeletePromptButton } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceDeletePromptButton';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rgPositivePromptChanged } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { AddPromptTriggerButton } from 'features/prompt/AddPromptTriggerButton';
import { PromptPopover } from 'features/prompt/PromptPopover';
import { usePrompt } from 'features/prompt/usePrompt';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const _focusVisible: SystemStyleObject = {
  outline: 'none',
};

export const RegionalGuidancePositivePrompt = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const selectPrompt = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => selectEntityOrThrow(canvas, entityIdentifier).positivePrompt ?? ''),
    [entityIdentifier]
  );
  const prompt = useAppSelector(selectPrompt);
  const dispatch = useAppDispatch();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useTranslation();
  const _onChange = useCallback(
    (v: string) => {
      dispatch(rgPositivePromptChanged({ entityIdentifier, prompt: v }));
    },
    [dispatch, entityIdentifier]
  );
  const onDeletePrompt = useCallback(() => {
    dispatch(rgPositivePromptChanged({ entityIdentifier, prompt: null }));
  }, [dispatch, entityIdentifier]);
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
          variant="outline"
          paddingInlineStart={2}
          paddingInlineEnd={8}
          minH={28}
          zIndex="0 !important"
          _focusVisible={_focusVisible}
        />
        <Flex
          flexDir="column"
          gap={2}
          position="absolute"
          insetBlockStart={2}
          insetInlineEnd={0}
          alignItems="center"
          justifyContent="center"
        >
          <RegionalGuidanceDeletePromptButton onDelete={onDeletePrompt} />
          <AddPromptTriggerButton isOpen={isOpen} onOpen={onOpen} />
        </Flex>
      </Box>
    </PromptPopover>
  );
});

RegionalGuidancePositivePrompt.displayName = 'RegionalGuidancePositivePrompt';
