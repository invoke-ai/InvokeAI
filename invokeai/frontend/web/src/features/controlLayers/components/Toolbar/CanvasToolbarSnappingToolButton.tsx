import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { selectSnapToGrid, settingsSnapToGridToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGridFourBold } from 'react-icons/pi';

export const CanvasToolbarSnappingToolButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const snapToGrid = useAppSelector(selectSnapToGrid);

  const onClick = useCallback(() => {
    dispatch(settingsSnapToGridToggled());
  }, [dispatch]);

  return (
    <IconButton
      onClick={onClick}
      variant="link"
      alignSelf="stretch"
      colorScheme={snapToGrid ? 'invokeBlue' : 'gray'}
      aria-label={t('controlLayers.settings.snapToGrid.label')}
      tooltip={t('controlLayers.settings.snapToGrid.label')}
      icon={<PiGridFourBold />}
      isDisabled={isBusy}
    />
  );
});

CanvasToolbarSnappingToolButton.displayName = 'CanvasToolbarSnappingToolButton';
