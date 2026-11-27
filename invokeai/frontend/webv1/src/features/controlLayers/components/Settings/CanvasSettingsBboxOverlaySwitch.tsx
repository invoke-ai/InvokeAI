import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectBboxOverlay, settingsBboxOverlayToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsBboxOverlaySwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const bboxOverlay = useAppSelector(selectBboxOverlay);
  const onChange = useCallback(() => {
    dispatch(settingsBboxOverlayToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.bboxOverlay')}
      </FormLabel>
      <Switch size="sm" isChecked={bboxOverlay} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsBboxOverlaySwitch.displayName = 'CanvasSettingsBboxOverlaySwitch';
