import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import type { InvIconButtonProps } from 'common/components/InvIconButton/types';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSync } from 'react-icons/fa';

import { useSyncModels } from './useSyncModels';

export const SyncModelsIconButton = memo(
  (props: Omit<InvIconButtonProps, 'aria-label'>) => {
    const { t } = useTranslation();
    const { syncModels, isLoading } = useSyncModels();
    const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

    if (!isSyncModelEnabled) {
      return null;
    }

    return (
      <InvIconButton
        icon={<FaSync />}
        tooltip={t('modelManager.syncModels')}
        aria-label={t('modelManager.syncModels')}
        isLoading={isLoading}
        onClick={syncModels}
        size="sm"
        variant="ghost"
        {...props}
      />
    );
  }
);

SyncModelsIconButton.displayName = 'SyncModelsIconButton';
