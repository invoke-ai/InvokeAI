import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxDimensionsSwapped } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsDownUpBold } from 'react-icons/pi';

export const BboxSwapDimensionsButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(selectIsStaging);
  const onClick = useCallback(() => {
    dispatch(bboxDimensionsSwapped());
  }, [dispatch]);
  return (
    <IconButton
      tooltip={t('parameters.swapDimensions')}
      aria-label={t('parameters.swapDimensions')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<PiArrowsDownUpBold />}
      isDisabled={isStaging}
    />
  );
});

BboxSwapDimensionsButton.displayName = 'BboxSwapDimensionsButton';
