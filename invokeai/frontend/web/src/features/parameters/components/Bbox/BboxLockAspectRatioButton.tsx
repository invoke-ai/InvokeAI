import { IconButton } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxAspectRatioLockToggled } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenBold } from 'react-icons/pi';

const selectAspectRatioIsLocked = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.aspectRatio.isLocked);

export const BboxLockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isLocked = useAppSelector(selectAspectRatioIsLocked);
  const isStaging = useAppSelector(selectIsStaging);
  const onClick = useCallback(() => {
    dispatch(bboxAspectRatioLockToggled());
  }, [dispatch]);

  return (
    <IconButton
      tooltip={t('parameters.lockAspectRatio')}
      aria-label={t('parameters.lockAspectRatio')}
      onClick={onClick}
      variant={isLocked ? 'outline' : 'ghost'}
      size="sm"
      icon={isLocked ? <PiLockSimpleFill /> : <PiLockSimpleOpenBold />}
      isDisabled={isStaging}
    />
  );
});

BboxLockAspectRatioButton.displayName = 'BboxLockAspectRatioButton';
