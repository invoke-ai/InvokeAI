import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectInvertScrollForToolWidth,
  settingsInvertScrollForToolWidthChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsInvertScrollCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const invertScrollForToolWidth = useAppSelector((state) => selectInvertScrollForToolWidth(state));
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(settingsInvertScrollForToolWidthChanged(e.target.checked));
    },
    [dispatch]
  );
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.settings.invertBrushSizeScrollDirection')}</FormLabel>
      <Checkbox isChecked={invertScrollForToolWidth} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsInvertScrollCheckbox.displayName = 'CanvasSettingsInvertScrollCheckbox';
