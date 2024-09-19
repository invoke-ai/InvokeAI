import { IconButton } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxSizeOptimized } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectCanvasSlice, selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { getIsSizeTooLarge, getIsSizeTooSmall } from 'features/parameters/util/optimalDimension';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleFill } from 'react-icons/pi';

const selectWidth = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.rect.width);
const selectHeight = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.rect.height);

export const BboxSetOptimalSizeButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(selectIsStaging);
  const width = useAppSelector(selectWidth);
  const height = useAppSelector(selectHeight);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isSizeTooSmall = useMemo(
    () => getIsSizeTooSmall(width, height, optimalDimension),
    [height, width, optimalDimension]
  );
  const isSizeTooLarge = useMemo(
    () => getIsSizeTooLarge(width, height, optimalDimension),
    [height, width, optimalDimension]
  );
  const onClick = useCallback(() => {
    dispatch(bboxSizeOptimized());
  }, [dispatch]);
  const tooltip = useMemo(() => {
    if (isSizeTooSmall) {
      return t('parameters.setToOptimalSizeTooSmall');
    }
    if (isSizeTooLarge) {
      return t('parameters.setToOptimalSizeTooLarge');
    }
    return t('parameters.setToOptimalSize');
  }, [isSizeTooLarge, isSizeTooSmall, t]);

  return (
    <IconButton
      tooltip={tooltip}
      aria-label={t('parameters.setToOptimalSize')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<PiSparkleFill />}
      colorScheme={isSizeTooSmall || isSizeTooLarge ? 'warning' : 'base'}
      isDisabled={isStaging}
    />
  );
});

BboxSetOptimalSizeButton.displayName = 'BboxSetOptimalSizeButton';
