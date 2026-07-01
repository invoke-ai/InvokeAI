import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectShowProgressOnCanvas,
  settingsShowProgressOnCanvasToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsShowProgressOnCanvas = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const showProgressOnCanvas = useAppSelector(selectShowProgressOnCanvas);
  const onChange = useCallback(() => {
    dispatch(settingsShowProgressOnCanvasToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.showProgressOnCanvas')}
      </FormLabel>
      <Switch size="sm" isChecked={showProgressOnCanvas} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsShowProgressOnCanvas.displayName = 'CanvasSettingsShowProgressOnCanvas';
