import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsClockwiseBold } from 'react-icons/pi';

import { useSyncModels } from './useSyncModels';

export const SyncModelsIconButton = memo((props: Omit<IconButtonProps, 'aria-label'>) => {
  const { t } = useTranslation();
  const { syncModels, isLoading } = useSyncModels();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

  if (!isSyncModelEnabled) {
    return null;
  }

  return (
    <IconButton
      icon={<PiArrowsClockwiseBold />}
      tooltip={t('modelManager.syncModels')}
      aria-label={t('modelManager.syncModels')}
      isLoading={isLoading}
      onClick={syncModels}
      size="sm"
      variant="ghost"
      {...props}
    />
  );
});

SyncModelsIconButton.displayName = 'SyncModelsIconButton';
