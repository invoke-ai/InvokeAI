import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { invertScrollChanged } from 'features/controlLayers/store/toolSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsInvertScrollCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const invertScroll = useAppSelector((s) => s.tool.invertScroll);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(invertScrollChanged(e.target.checked)),
    [dispatch]
  );
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('unifiedCanvas.invertBrushSizeScrollDirection')}</FormLabel>
      <Checkbox isChecked={invertScroll} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsInvertScrollCheckbox.displayName = 'CanvasSettingsInvertScrollCheckbox';
