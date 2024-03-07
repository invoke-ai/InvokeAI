import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsClockwiseBold } from 'react-icons/pi';

import { useSyncModels } from './useSyncModels';

export const SyncModelsButton = memo((props: Omit<ButtonProps, 'aria-label'>) => {
  const { t } = useTranslation();
  const { syncModels, isLoading } = useSyncModels();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

  if (!isSyncModelEnabled) {
    return null;
  }

  return (
    <Button
      leftIcon={<PiArrowsClockwiseBold />}
      isLoading={isLoading}
      onClick={syncModels}
      variant="ghost"
      {...props}
    >
      {t('modelManager.syncModels')}
    </Button>
  );
});

SyncModelsButton.displayName = 'SyncModelsButton';
