import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { aspectRatioLockToggled, selectAspectRatioIsLocked } from 'features/controlLayers/store/paramsSlice';
import { useIsApiModel } from 'features/parameters/hooks/useIsApiModel';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenBold } from 'react-icons/pi';

export const DimensionsLockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isLocked = useAppSelector(selectAspectRatioIsLocked);
  const isApiModel = useIsApiModel();

  const onClick = useCallback(() => {
    dispatch(aspectRatioLockToggled());
  }, [dispatch]);

  return (
    <IconButton
      tooltip={t('parameters.lockAspectRatio')}
      aria-label={t('parameters.lockAspectRatio')}
      onClick={onClick}
      variant={isLocked ? 'outline' : 'ghost'}
      size="sm"
      icon={isLocked ? <PiLockSimpleFill /> : <PiLockSimpleOpenBold />}
      isDisabled={isApiModel}
    />
  );
});

DimensionsLockAspectRatioButton.displayName = 'DimensionsLockAspectRatioButton';
