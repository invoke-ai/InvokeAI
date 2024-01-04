import { useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import {
  getIsSizeTooLarge,
  getIsSizeTooSmall,
} from 'features/parameters/util/optimalDimension';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { IoSparkles } from 'react-icons/io5';

export const SetOptimalSizeButton = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isSizeTooSmall = useMemo(
    () => getIsSizeTooSmall(ctx.width, ctx.height, optimalDimension),
    [ctx.height, ctx.width, optimalDimension]
  );
  const isSizeTooLarge = useMemo(
    () => getIsSizeTooLarge(ctx.width, ctx.height, optimalDimension),
    [ctx.height, ctx.width, optimalDimension]
  );
  const onClick = useCallback(() => {
    ctx.setOptimalSize();
  }, [ctx]);
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
    <InvIconButton
      tooltip={tooltip}
      aria-label={t('parameters.setToOptimalSize')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<IoSparkles />}
      colorScheme={isSizeTooSmall || isSizeTooLarge ? 'warning' : 'base'}
    />
  );
});

SetOptimalSizeButton.displayName = 'SetOptimalSizeButton';
