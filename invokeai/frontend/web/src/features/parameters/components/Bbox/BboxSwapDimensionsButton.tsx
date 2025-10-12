import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasIsStaging } from 'features/controlLayers/hooks/useCanvasIsStaging';
import { bboxDimensionsSwapped } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsDownUpBold } from 'react-icons/pi';

export const BboxSwapDimensionsButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isStaging = useCanvasIsStaging();
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
