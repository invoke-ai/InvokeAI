import { IconButton } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxAspectRatioLockToggled } from 'features/controlLayers/store/canvasInstanceSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenBold } from 'react-icons/pi';

const selectAspectRatioIsLocked = createSelector(selectCanvasSlice, (canvas) => canvas ? canvas.bbox.aspectRatio.isLocked : false);

export const BboxLockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isLocked = useAppSelector(selectAspectRatioIsLocked);
  const isBboxSizeLocked = useIsBboxSizeLocked();

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
      isDisabled={isBboxSizeLocked}
    />
  );
});

BboxLockAspectRatioButton.displayName = 'BboxLockAspectRatioButton';
