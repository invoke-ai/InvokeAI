import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectPressureAffectsOpacity,
  selectPressureAffectsWidth,
  settingsPressureAffectsOpacityToggled,
  settingsPressureAffectsWidthToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEventHandler } from 'react';
import { Fragment, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsPressureOptions = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const pressureAffectsWidth = useAppSelector(selectPressureAffectsWidth);
  const pressureAffectsOpacity = useAppSelector(selectPressureAffectsOpacity);
  const onWidthChange = useCallback<ChangeEventHandler<HTMLInputElement>>(() => {
    dispatch(settingsPressureAffectsWidthToggled());
  }, [dispatch]);
  const onOpacityChange = useCallback<ChangeEventHandler<HTMLInputElement>>(() => {
    dispatch(settingsPressureAffectsOpacityToggled());
  }, [dispatch]);

  return (
    <Fragment>
      <FormControl w="full">
        <FormLabel flexGrow={1}>{t('controlLayers.settings.pressureAffectsWidth')}</FormLabel>
        <Checkbox isChecked={pressureAffectsWidth} onChange={onWidthChange} />
      </FormControl>
      <FormControl w="full">
        <FormLabel flexGrow={1}>{t('controlLayers.settings.pressureAffectsBrushOpacity')}</FormLabel>
        <Checkbox isChecked={pressureAffectsOpacity} onChange={onOpacityChange} />
      </FormControl>
    </Fragment>
  );
});

CanvasSettingsPressureOptions.displayName = 'CanvasSettingsPressureOptions';
