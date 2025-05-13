import { Box, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { PromptLabel } from 'features/parameters/components/Prompts/PromptLabel';
import { positivePromptChanged, selectPositivePrompt } from 'features/simpleGeneration/store/simpleGenerationSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SimpleTabPositivePrompt = memo(() => {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectPositivePrompt);

  const { t } = useTranslation();
  const onChange = useCallback<ChangeEventHandler<HTMLTextAreaElement>>(
    (e) => {
      dispatch(positivePromptChanged({ positivePrompt: e.target.value }));
    },
    [dispatch]
  );

  useRegisteredHotkeys({
    id: 'focusPrompt',
    category: 'app',
    callback: focus,
    options: { preventDefault: true, enableOnFormTags: ['INPUT', 'SELECT', 'TEXTAREA'] },
    dependencies: [focus],
  });

  return (
    <Box pos="relative">
      <Textarea
        id="prompt"
        name="prompt"
        value={prompt}
        onChange={onChange}
        minH={40}
        variant="outline"
        paddingInlineEnd={10}
        paddingInlineStart={3}
        paddingTop="24px"
        paddingBottom={3}
      />
      <PromptLabel label={t('parameters.positivePromptPlaceholder')} />
    </Box>
  );
});

SimpleTabPositivePrompt.displayName = 'SimpleTabPositivePrompt';
