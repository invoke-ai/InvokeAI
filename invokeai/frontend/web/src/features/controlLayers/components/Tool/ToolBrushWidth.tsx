import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { selectCanvasSettingsSlice, settingsBrushWidthChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';

import { ToolWidth } from './ToolWidth';

const selectBrushWidth = createSelector(selectCanvasSettingsSlice, (settings) => settings.brushWidth);

export const ToolBrushWidth = memo(() => {
  const dispatch = useAppDispatch();
  const isSelected = useToolIsSelected('brush');
  const width = useAppSelector(selectBrushWidth);

  const onValueChange = useCallback(
    (value: number) => {
      dispatch(settingsBrushWidthChanged(value));
    },
    [dispatch]
  );

  return <ToolWidth isSelected={isSelected} width={width} onValueChange={onValueChange} />;
});

ToolBrushWidth.displayName = 'ToolBrushWidth';
