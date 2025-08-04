import { Alert, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasAlertsBboxVisibility = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isBboxHidden = useStore(canvasManager.tool.tools.bbox.$isBboxHidden);

  if (!isBboxHidden) {
    return null;
  }

  return (
    <Alert status="warning" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{t('controlLayers.warnings.bboxHidden')}</AlertTitle>
    </Alert>
  );
});

CanvasAlertsBboxVisibility.displayName = 'CanvasAlertsBboxVisibility';
