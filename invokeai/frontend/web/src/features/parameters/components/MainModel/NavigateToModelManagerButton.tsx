import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $onClickGoToModelManager } from 'app/store/nanostores/onClickGoToModelManager';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsModelsTabDisabled } from 'features/system/store/configSlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCubeBold } from 'react-icons/pi';

export const NavigateToModelManagerButton = memo((props: Omit<IconButtonProps, 'aria-label'>) => {
  const isModelsTabDisabled = useAppSelector(selectIsModelsTabDisabled);
  const onClickGoToModelManager = useStore($onClickGoToModelManager);

  const { t } = useTranslation();

  const onClick = useCallback(() => {
    navigationApi.switchToTab('models');
  }, []);

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
