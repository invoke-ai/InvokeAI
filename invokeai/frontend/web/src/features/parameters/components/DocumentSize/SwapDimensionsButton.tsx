import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { documentDimensionsSwapped } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsDownUpBold } from 'react-icons/pi';

export const SwapDimensionsButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(documentDimensionsSwapped());
  }, [dispatch]);
  return (
    <IconButton
      tooltip={t('parameters.swapDimensions')}
      aria-label={t('parameters.swapDimensions')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<PiArrowsDownUpBold />}
    />
  );
});

SwapDimensionsButton.displayName = 'SwapDimensionsButton';
