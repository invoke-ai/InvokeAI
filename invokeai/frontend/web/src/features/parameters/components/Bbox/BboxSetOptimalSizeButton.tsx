import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxSizeOptimized } from 'features/controlLayers/store/canvasInstanceSlice';
import { selectHeight,selectOptimalDimension, selectWidth } from 'features/controlLayers/store/selectors';
import { useIsBboxSizeLocked } from 'features/parameters/components/Bbox/use-is-bbox-size-locked';
import { getIsSizeTooLarge, getIsSizeTooSmall } from 'features/parameters/util/optimalDimension';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleFill } from 'react-icons/pi';


export const BboxSetOptimalSizeButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBboxSizeLocked = useIsBboxSizeLocked();
  const width = useAppSelector(selectWidth);
  const height = useAppSelector(selectHeight);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const isSizeTooSmall = useMemo(
    () => width != null && height != null ? getIsSizeTooSmall(width, height, optimalDimension) : false,
    [height, width, optimalDimension]
  );
  const isSizeTooLarge = useMemo(
    () => width != null && height != null ? getIsSizeTooLarge(width, height, optimalDimension) : false,
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
      isDisabled={isBboxSizeLocked}
    />
  );
});

BboxSetOptimalSizeButton.displayName = 'BboxSetOptimalSizeButton';
