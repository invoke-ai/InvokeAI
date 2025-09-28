import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  aspectRatioLockToggled,
  selectAspectRatioIsLocked,
  selectIsApiBaseModel,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenBold } from 'react-icons/pi';

export const DimensionsLockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const dispatchParams = useParamsDispatch();
  const isLocked = useAppSelector(selectAspectRatioIsLocked);
  const isApiModel = useAppSelector(selectIsApiBaseModel);

  const onClick = useCallback(() => {
    dispatchParams(aspectRatioLockToggled);
  }, [dispatchParams]);

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
