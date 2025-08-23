import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useItemDTOContextImageOnly } from 'features/gallery/contexts/ItemDTOContext';
import { expandPrompt } from 'features/prompt/PromptExpansion/expand';
import { promptExpansionApi } from 'features/prompt/PromptExpansion/state';
import { selectAllowPromptExpansion } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTextTBold } from 'react-icons/pi';

export const ContextMenuItemUseForPromptGeneration = memo(() => {
  const { t } = useTranslation();
  const { dispatch, getState } = useAppStore();
  const imageDTO = useItemDTOContextImageOnly();
  const { isPending } = useStore(promptExpansionApi.$state);
  const isPromptExpansionEnabled = useAppSelector(selectAllowPromptExpansion);

  const handleUseForPromptGeneration = useCallback(() => {
    promptExpansionApi.setPending(imageDTO);
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

ContextMenuItemUseForPromptGeneration.displayName = 'ContextMenuItemUseForPromptGeneration';
