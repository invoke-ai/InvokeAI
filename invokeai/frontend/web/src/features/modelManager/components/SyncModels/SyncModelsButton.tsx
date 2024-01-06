import { InvButton } from 'common/components/InvButton/InvButton';
import type { InvButtonProps } from 'common/components/InvButton/types';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSync } from 'react-icons/fa';

import { useSyncModels } from './useSyncModels';

export const SyncModelsButton = memo(
  (props: Omit<InvButtonProps, 'children'>) => {
    const { t } = useTranslation();
    const { syncModels, isLoading } = useSyncModels();
    const isSyncModelEnabled = useFeatureStatus('syncModels').isFeatureEnabled;

    if (!isSyncModelEnabled) {
      return null;
    }

    return (
      <InvButton
        isLoading={isLoading}
        onClick={syncModels}
        leftIcon={<FaSync />}
        minW="max-content"
        {...props}
      >
        {t('modelManager.syncModels')}
      </InvButton>
    );
  }
);

SyncModelsButton.displayName = 'SyncModelsButton';
