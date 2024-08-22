import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { canvasReset } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsResetButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(canvasReset());
  }, [dispatch]);
  return (
    <Button onClick={onClick} colorScheme="error" size="sm">
      {t('controlLayers.resetCanvas')}
    </Button>
  );
});

CanvasSettingsResetButton.displayName = 'CanvasSettingsResetButton';
