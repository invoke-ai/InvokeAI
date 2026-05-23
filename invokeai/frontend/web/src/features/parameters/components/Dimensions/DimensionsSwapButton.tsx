import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { dimensionsSwapped, selectHasFixedDimensionSizes } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsDownUpBold } from 'react-icons/pi';

export const DimensionsSwapButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const hasFixedSizes = useAppSelector(selectHasFixedDimensionSizes);
  const onClick = useCallback(() => {
    dispatch(dimensionsSwapped());
  }, [dispatch]);
  return (
    <IconButton
      tooltip={t('parameters.swapDimensions')}
      aria-label={t('parameters.swapDimensions')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<PiArrowsDownUpBold />}
      isDisabled={hasFixedSizes}
    />
  );
});

DimensionsSwapButton.displayName = 'DimensionsSwapButton';
