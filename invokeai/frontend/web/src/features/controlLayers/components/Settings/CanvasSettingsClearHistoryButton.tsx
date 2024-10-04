import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { canvasClearHistory } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsClearHistoryButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onPointerUp = useCallback(() => {
    dispatch(canvasClearHistory());
  }, [dispatch]);
  return (
    <Button onPointerUp={onPointerUp} size="sm">
      {t('controlLayers.clearHistory')}
    </Button>
  );
});

CanvasSettingsClearHistoryButton.displayName = 'CanvasSettingsClearHistoryButton';
