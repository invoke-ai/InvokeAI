import { Button } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsLogDebugInfoButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const onClick = useCallback(() => {
    canvasManager.logDebugInfo();
  }, [canvasManager]);
  return (
    <Button onClick={onClick} size="sm">
      {t('controlLayers.logDebugInfo')}
    </Button>
  );
});

CanvasSettingsLogDebugInfoButton.displayName = 'CanvasSettingsLogDebugInfoButton';
