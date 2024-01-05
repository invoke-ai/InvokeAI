import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaLock, FaLockOpen } from 'react-icons/fa6';

export const LockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const onClick = useCallback(() => {
    ctx.isLockedToggled();
  }, [ctx]);

  return (
    <InvIconButton
      tooltip={t('parameters.lockAspectRatio')}
      aria-label={t('parameters.lockAspectRatio')}
      onClick={onClick}
      variant={ctx.aspectRatioState.isLocked ? 'outline' : 'ghost'}
      size="sm"
      icon={ctx.aspectRatioState.isLocked ? <FaLock /> : <FaLockOpen />}
    />
  );
});

LockAspectRatioButton.displayName = 'LockAspectRatioButton';
