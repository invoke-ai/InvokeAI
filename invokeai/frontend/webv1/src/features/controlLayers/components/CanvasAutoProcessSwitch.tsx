import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAutoProcess, settingsAutoProcessToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasAutoProcessSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoProcess = useAppSelector(selectAutoProcess);

  const onChange = useCallback(() => {
    dispatch(settingsAutoProcessToggled());
  }, [dispatch]);

  return (
    <FormControl w="min-content">
      <FormLabel m={0}>{t('controlLayers.filter.autoProcess')}</FormLabel>
      <Switch size="sm" isChecked={autoProcess} onChange={onChange} />
    </FormControl>
  );
});

CanvasAutoProcessSwitch.displayName = 'CanvasAutoProcessSwitch';
