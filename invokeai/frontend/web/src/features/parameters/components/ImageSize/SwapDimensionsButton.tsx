import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { IoSwapVertical } from 'react-icons/io5';

export const SwapDimensionsButton = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const onClick = useCallback(() => {
    ctx.dimensionsSwapped();
  }, [ctx]);
  return (
    <InvIconButton
      aria-label={t('parameters.swapDimensions')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<IoSwapVertical />}
    />
  );
});

SwapDimensionsButton.displayName = 'SwapDimensionsButton';
