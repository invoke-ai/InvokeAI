import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $onClickGoToModelManager } from 'app/store/nanostores/onClickGoToModelManager';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectIsModelsTabDisabled } from 'features/system/store/configSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCubeBold } from 'react-icons/pi';

export const NavigateToModelManagerButton = memo((props: Omit<IconButtonProps, 'aria-label'>) => {
  const isModelsTabDisabled = useAppSelector(selectIsModelsTabDisabled);
  const onClickGoToModelManager = useStore($onClickGoToModelManager);

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(setActiveTab('models'));
  }, [dispatch]);

  if (isModelsTabDisabled && !onClickGoToModelManager) {
    return null;
  }

  return (
    <IconButton
      icon={<PiCubeBold />}
      tooltip={`${t('modelManager.manageModels')}`}
      aria-label={`${t('modelManager.manageModels')}`}
      onClick={onClickGoToModelManager ?? onClick}
      size="sm"
      variant="ghost"
      {...props}
    />
  );
});

NavigateToModelManagerButton.displayName = 'NavigateToModelManagerButton';
