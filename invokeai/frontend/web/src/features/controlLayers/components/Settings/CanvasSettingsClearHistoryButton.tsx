import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { canvasClearHistory } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsClearHistoryButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(canvasClearHistory());
  }, [dispatch]);
  return (
    <Button onClick={onClick} size="sm">
      {t('controlLayers.clearHistory')}
    </Button>
  );
});

CanvasSettingsClearHistoryButton.displayName = 'CanvasSettingsClearHistoryButton';
