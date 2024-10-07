import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectIsModelsTabDisabled } from 'features/system/store/configSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';

export const NavigateToModelManagerButton = memo((props: Omit<IconButtonProps, 'aria-label'>) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isModelsTabDisabled = useAppSelector(selectIsModelsTabDisabled);

  const handleClick = useCallback(() => {
    dispatch(setActiveTab('models'));
  }, [dispatch]);

  if (isModelsTabDisabled) {
    return null;
  }

  return (
    <IconButton
      icon={<PiGearSixFill />}
      tooltip={`${t('common.goTo')} ${t('ui.tabs.modelsTab')}`}
      aria-label={`${t('common.goTo')} ${t('ui.tabs.modelsTab')}`}
      onClick={handleClick}
      size="sm"
      variant="ghost"
      {...props}
    />
  );
});

NavigateToModelManagerButton.displayName = 'NavigateToModelManagerButton';
