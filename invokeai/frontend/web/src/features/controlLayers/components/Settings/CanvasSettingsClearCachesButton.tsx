import { Button } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsClearCachesButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const clearCaches = useCallback(() => {
    canvasManager.cache.clearAll();
  }, [canvasManager]);
  return (
    <Button onClick={clearCaches} size="sm" colorScheme="warning">
      {t('controlLayers.clearCaches')}
    </Button>
  );
});

CanvasSettingsClearCachesButton.displayName = 'CanvasSettingsClearCachesButton';
