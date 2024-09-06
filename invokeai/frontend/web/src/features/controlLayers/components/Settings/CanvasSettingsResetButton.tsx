import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { canvasReset } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsResetButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const onClick = useCallback(() => {
    dispatch(canvasReset());
    canvasManager.stage.fitLayersToStage();
  }, [canvasManager.stage, dispatch]);
  return (
    <Button onClick={onClick} colorScheme="error" size="sm">
      {t('controlLayers.resetCanvas')}
    </Button>
  );
});

CanvasSettingsResetButton.displayName = 'CanvasSettingsResetButton';
