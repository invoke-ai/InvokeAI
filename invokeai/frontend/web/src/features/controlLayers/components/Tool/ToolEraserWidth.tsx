import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import {
  selectCanvasSettingsSlice,
  settingsEraserWidthChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';

import { ToolWidth } from './ToolWidth';

const selectEraserWidth = createSelector(selectCanvasSettingsSlice, (settings) => settings.eraserWidth);

export const ToolEraserWidth = memo(() => {
  const dispatch = useAppDispatch();
  const isSelected = useToolIsSelected('eraser');
  const width = useAppSelector(selectEraserWidth);

  const onValueChange = useCallback(
    (value: number) => {
      dispatch(settingsEraserWidthChanged(value));
    },
    [dispatch]
  );

  return <ToolWidth isSelected={isSelected} width={width} onValueChange={onValueChange} />;
});

ToolEraserWidth.displayName = 'ToolEraserWidth';
