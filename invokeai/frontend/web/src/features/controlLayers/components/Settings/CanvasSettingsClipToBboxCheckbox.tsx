import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clipToBboxChanged } from 'features/controlLayers/store/canvasV2Slice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsClipToBboxCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const clipToBbox = useAppSelector((s) => s.canvasV2.settings.clipToBbox);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(clipToBboxChanged(e.target.checked)),
    [dispatch]
  );
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.clipToBbox')}</FormLabel>
      <Checkbox isChecked={clipToBbox} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsClipToBboxCheckbox.displayName = 'CanvasSettingsClipToBboxCheckbox';
