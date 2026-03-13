import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { aspectRatioLockToggled, selectAspectRatioIsLocked } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenBold } from 'react-icons/pi';

export const DimensionsLockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isLocked = useAppSelector(selectAspectRatioIsLocked);

  const onClick = useCallback(() => {
    dispatch(aspectRatioLockToggled());
  }, [dispatch]);

  return (
    <IAITooltip label={t('parameters.lockAspectRatio')}>
      <IconButton
        aria-label={t('parameters.lockAspectRatio')}
        onClick={onClick}
        variant={isLocked ? 'outline' : 'ghost'}
        size="sm"
        icon={isLocked ? <PiLockSimpleFill /> : <PiLockSimpleOpenBold />}
      />
    </IAITooltip>
  );
});

DimensionsLockAspectRatioButton.displayName = 'DimensionsLockAspectRatioButton';
