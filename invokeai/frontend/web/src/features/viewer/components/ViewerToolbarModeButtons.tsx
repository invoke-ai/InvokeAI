import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { ButtonGroup, IconButton, spinAnimation } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsProcessing } from 'features/queue/hooks/useIsProcessing';
import { viewerModeChanged } from 'features/viewer/store/viewerSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCircleNotchBold, PiEyeBold, PiHourglassMediumFill, PiInfoBold } from 'react-icons/pi';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const ViewerToolbarModeButtons = memo(() => {
  const dispatch = useAppDispatch();
  const isProcessing = useIsProcessing();
  const viewerMode = useAppSelector((s) => s.viewer.viewerMode);
  const { t } = useTranslation();

  const handleSelectViewerImage = useCallback(() => {
    dispatch(viewerModeChanged('image'));
  }, [dispatch]);
  // TODO: hotkey

  const handleSelectViewerInfo = useCallback(() => {
    dispatch(viewerModeChanged('info'));
  }, [dispatch]);
  useHotkeys('i', handleSelectViewerInfo, [handleSelectViewerInfo]);

  const handleSelectViewerProgress = useCallback(() => {
    dispatch(viewerModeChanged('progress'));
  }, [dispatch]);
  // TODO: hotkey

  return (
    <ButtonGroup>
      <IconButton
        icon={<PiEyeBold />}
        tooltip={`${t('viewer.viewerModeImage')}`}
        aria-label={`${t('viewer.viewerModeImage')}`}
        isChecked={viewerMode === 'image'}
        onClick={handleSelectViewerImage}
      />
      <IconButton
        icon={<PiInfoBold />}
        tooltip={`${t('viewer.viewerModeInfo')} (I)`}
        aria-label={`${t('viewer.viewerModeInfo')} (I)`}
        isChecked={viewerMode === 'info'}
        onClick={handleSelectViewerInfo}
      />
      <IconButton
        aria-label={`${t('viewer.viewerModeProgress')}`}
        tooltip={`${t('viewer.viewerModeProgress')}`}
        icon={isProcessing ? <PiCircleNotchBold /> : <PiHourglassMediumFill />}
        isChecked={viewerMode === 'progress'}
        onClick={handleSelectViewerProgress}
        sx={isProcessing ? loadingStyles : undefined}
      />
    </ButtonGroup>
  );
});

ViewerToolbarModeButtons.displayName = 'ViewerToolbarModeButtons';
