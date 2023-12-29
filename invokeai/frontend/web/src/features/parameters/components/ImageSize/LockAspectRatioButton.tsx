import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { isLockedToggled } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaLock, FaLockOpen } from 'react-icons/fa6';

export const LockAspectRatioButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isLocked = useAppSelector(
    (state) => state.generation.aspectRatio.isLocked
  );
  const onClick = useCallback(() => {
    dispatch(isLockedToggled());
  }, [dispatch]);

  return (
    <InvIconButton
      aria-label={t('parameters.lockAspectRatio')}
      onClick={onClick}
      variant={isLocked ? 'outline' : 'ghost'}
      size="sm"
      icon={isLocked ? <FaLock /> : <FaLockOpen />}
    />
  );
});

LockAspectRatioButton.displayName = 'LockAspectRatioButton';
