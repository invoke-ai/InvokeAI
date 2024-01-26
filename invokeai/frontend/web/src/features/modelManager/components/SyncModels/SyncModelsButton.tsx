import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsClockwiseBold } from 'react-icons/pi';

import { useSyncModels } from './useSyncModels';

export const SyncModelsButton = memo((props: Omit<ButtonProps, 'children'>) => {
  const { t } = useTranslation();
  const { syncModels, isLoading } = useSyncModels();
  const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

  if (!isSyncModelEnabled) {
    return null;
  }

  return (
    <Button
      isLoading={isLoading}
      onClick={syncModels}
      leftIcon={<PiArrowsClockwiseBold />}
      minW="max-content"
      {...props}
    >
      {t('modelManager.syncModels')}
    </Button>
  );
});

SyncModelsButton.displayName = 'SyncModelsButton';
