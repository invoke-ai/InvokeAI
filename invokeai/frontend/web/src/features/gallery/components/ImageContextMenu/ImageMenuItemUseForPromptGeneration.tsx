import { MenuItem } from '@invoke-ai/ui-library';
import { promptGenerationFromImageRequested } from 'app/store/middleware/listenerMiddleware/listeners/addPromptExpansionRequestedListener';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTextTBold } from 'react-icons/pi';

export const ImageMenuItemUseForPromptGeneration = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const handleUseForPromptGeneration = useCallback(() => {
    dispatch(promptGenerationFromImageRequested({ imageDTO }));
    toast({
      id: 'PROMPT_GENERATION_STARTED',
      title: t('toast.promptGenerationStarted'),
      status: 'info',
    });
  }, [dispatch, imageDTO, t]);

  return (
    <MenuItem icon={<PiTextTBold />} onClickCapture={handleUseForPromptGeneration} id="use-for-prompt-generation">
      {t('gallery.useForPromptGeneration')}
    </MenuItem>
  );
});

ImageMenuItemUseForPromptGeneration.displayName = 'ImageMenuItemUseForPromptGeneration';
