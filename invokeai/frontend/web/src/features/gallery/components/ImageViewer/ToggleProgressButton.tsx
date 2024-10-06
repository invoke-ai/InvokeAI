import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { setShouldShowProgressInViewer } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHourglassHighBold } from 'react-icons/pi';

export const ToggleProgressButton = memo(() => {
  const dispatch = useAppDispatch();
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    dispatch(setShouldShowProgressInViewer(!shouldShowProgressInViewer));
  }, [dispatch, shouldShowProgressInViewer]);

  return (
    <IconButton
      aria-label={t('settings.displayInProgress')}
      tooltip={t('settings.displayInProgress')}
      icon={<PiHourglassHighBold />}
      onClick={onClick}
      variant="outline"
      colorScheme={shouldShowProgressInViewer ? 'invokeBlue' : 'base'}
      data-testid="toggle-show-progress-button"
    />
  );
});

ToggleProgressButton.displayName = 'ToggleProgressButton';
