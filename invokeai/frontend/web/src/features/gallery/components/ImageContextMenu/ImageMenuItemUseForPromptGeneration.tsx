import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { expandPrompt } from 'features/prompt/PromptExpansion/expand';
import { promptExpansionApi } from 'features/prompt/PromptExpansion/state';
import { selectAllowPromptExpansion } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTextTBold } from 'react-icons/pi';

export const ImageMenuItemUseForPromptGeneration = memo(() => {
  const { t } = useTranslation();
  const { dispatch, getState } = useAppStore();
  const imageDTO = useImageDTOContext();
  const { isPending } = useStore(promptExpansionApi.$state);
  const isPromptExpansionEnabled = useAppSelector(selectAllowPromptExpansion);

  const handleUseForPromptGeneration = useCallback(() => {
    expandPrompt({ dispatch, getState, imageDTO });
    toast({
      id: 'PROMPT_GENERATION_STARTED',
      title: t('toast.promptGenerationStarted'),
      status: 'info',
    });
  }, [dispatch, getState, imageDTO, t]);

  if (!isPromptExpansionEnabled) {
    return null;
  }

  return (
    <MenuItem
      icon={<PiTextTBold />}
      onClickCapture={handleUseForPromptGeneration}
      id="use-for-prompt-generation"
      isDisabled={isPending}
    >
      {t('gallery.useForPromptGeneration')}
    </MenuItem>
  );
});

ImageMenuItemUseForPromptGeneration.displayName = 'ImageMenuItemUseForPromptGeneration';
