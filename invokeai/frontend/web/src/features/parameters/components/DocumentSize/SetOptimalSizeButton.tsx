import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { documentSizeOptimized } from 'features/controlLayers/store/canvasV2Slice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { getIsSizeTooLarge, getIsSizeTooSmall } from 'features/parameters/util/optimalDimension';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSparklingFill } from 'react-icons/ri';

export const SetOptimalSizeButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const width = useAppSelector((s) => s.canvasV2.document.width);
  const height = useAppSelector((s) => s.canvasV2.document.height);
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
    dispatch(documentSizeOptimized());
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
      icon={<RiSparklingFill />}
      colorScheme={isSizeTooSmall || isSizeTooLarge ? 'warning' : 'base'}
    />
  );
});

SetOptimalSizeButton.displayName = 'SetOptimalSizeButton';
