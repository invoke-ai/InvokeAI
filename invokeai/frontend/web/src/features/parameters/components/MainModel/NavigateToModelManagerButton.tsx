import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixBold } from 'react-icons/pi';

export const NavigateToModelManagerButton = memo((props: Omit<IconButtonProps, 'aria-label'>) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const shouldShowButton = useMemo(() => !disabledTabs.includes('modelManager'), [disabledTabs]);

  const handleClick = useCallback(() => {
    dispatch(setActiveTab('modelManager'));
  }, [dispatch]);

  if (!shouldShowButton) {
    return null;
  }

  return (
    <IconButton
      icon={<PiGearSixBold />}
      tooltip={t('modelManager.modelManager')}
      aria-label={t('modelManager.modelManager')}
      onClick={handleClick}
      size="sm"
      variant="ghost"
      {...props}
    />
  );
});

NavigateToModelManagerButton.displayName = 'NavigateToModelManagerButton';
