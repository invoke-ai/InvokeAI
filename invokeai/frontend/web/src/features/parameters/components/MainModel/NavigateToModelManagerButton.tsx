import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCubeBold } from 'react-icons/pi';

export const NavigateToModelManagerButton = memo((props: Omit<IconButtonProps, 'aria-label'>) => {
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    navigationApi.switchToTab('models');
  }, []);

  return (
    <IconButton
      icon={<PiCubeBold />}
      tooltip={`${t('modelManager.manageModels')}`}
      aria-label={`${t('modelManager.manageModels')}`}
      onClick={onClick}
      size="sm"
      variant="ghost"
      {...props}
    />
  );
});

NavigateToModelManagerButton.displayName = 'NavigateToModelManagerButton';
