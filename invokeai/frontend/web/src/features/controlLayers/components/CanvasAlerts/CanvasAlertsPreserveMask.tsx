import { Alert, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectPreserveMask } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasAlertsPreserveMask = memo(() => {
  const { t } = useTranslation();
  const preserveMask = useAppSelector((state) => selectPreserveMask(state));

  if (!preserveMask) {
    return null;
  }

  return (
    <Alert status="warning" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{t('controlLayers.settings.preserveMask.alert')}</AlertTitle>
    </Alert>
  );
});

CanvasAlertsPreserveMask.displayName = 'CanvasAlertsPreserveMask';
