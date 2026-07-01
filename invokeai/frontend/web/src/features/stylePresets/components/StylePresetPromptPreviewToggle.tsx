import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShowPromptPreviews, showPromptPreviewsChanged } from 'features/stylePresets/store/stylePresetSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiEyeSlashBold } from 'react-icons/pi';

export const StylePresetPromptPreviewToggle = () => {
  const dispatch = useAppDispatch();
  const showPromptPreviews = useAppSelector(selectShowPromptPreviews);
  const { t } = useTranslation();

  const handleToggle = useCallback(() => {
    dispatch(showPromptPreviewsChanged(!showPromptPreviews));
  }, [dispatch, showPromptPreviews]);

  return (
    <IconButton
      size="sm"
      variant="link"
      alignSelf="stretch"
      aria-label={t('stylePresets.togglePromptPreviews')}
      tooltip={t('stylePresets.togglePromptPreviews')}
      onClick={handleToggle}
      icon={showPromptPreviews ? <PiEyeBold /> : <PiEyeSlashBold />}
      colorScheme={showPromptPreviews ? 'invokeBlue' : 'base'}
    />
  );
};
