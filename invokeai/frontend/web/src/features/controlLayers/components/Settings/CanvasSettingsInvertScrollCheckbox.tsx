import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { invertScrollChanged, selectToolSlice } from 'features/controlLayers/store/toolSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectInvertScroll = createSelector(selectToolSlice, (tool) => tool.invertScroll);

export const CanvasSettingsInvertScrollCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const invertScroll = useAppSelector(selectInvertScroll);
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
