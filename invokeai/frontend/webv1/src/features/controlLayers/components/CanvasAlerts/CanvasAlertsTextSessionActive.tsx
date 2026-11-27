import { Alert, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasAlertsTextSessionActive = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const session = useStore(canvasManager.tool.tools.text.$session);

  if (!session || session.status === 'committed') {
    return null;
  }

  return (
    <Alert status="warning" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{t('controlLayers.HUD.textSessionActive')}</AlertTitle>
    </Alert>
  );
});

CanvasAlertsTextSessionActive.displayName = 'CanvasAlertsTextSessionActive';
