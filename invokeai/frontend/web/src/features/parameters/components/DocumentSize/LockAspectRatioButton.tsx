import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { documentAspectRatioLockToggled } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenBold } from 'react-icons/pi';

export const LockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isLocked = useAppSelector((s) => s.canvasV2.document.aspectRatio.isLocked);
  const onClick = useCallback(() => {
    dispatch(documentAspectRatioLockToggled());
  }, [dispatch]);

  return (
    <IconButton
      tooltip={t('parameters.lockAspectRatio')}
      aria-label={t('parameters.lockAspectRatio')}
      onClick={onClick}
      variant={isLocked ? 'outline' : 'ghost'}
      size="sm"
      icon={isLocked ? <PiLockSimpleFill /> : <PiLockSimpleOpenBold />}
    />
  );
});

LockAspectRatioButton.displayName = 'LockAspectRatioButton';
