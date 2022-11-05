import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaPaintBrush } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import IAINumberInput from 'common/components/IAINumberInput';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  // InpaintingState,
  setBrushSize,
  setShouldShowBrushPreview,
  setTool,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import IAICanvasMaskColorPicker from './IAICanvasMaskControls/IAICanvasMaskColorPicker';

const inpaintingBrushSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { tool, brushSize, shouldShowMask } = currentCanvas;

    return {
      tool,
      brushSize,
      shouldShowMask,
      activeTabName,
      areHotkeysEnabled,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasBrushControl() {
  const dispatch = useAppDispatch();
  const { tool, brushSize, shouldShowMask, activeTabName, areHotkeysEnabled } =
    useAppSelector(inpaintingBrushSelector);

  const handleSelectBrushTool = () => dispatch(setTool('brush'));

  const handleShowBrushPreview = () => {
    dispatch(setShouldShowBrushPreview(true));
  };

  const handleHideBrushPreview = () => {
    dispatch(setShouldShowBrushPreview(false));
  };

  const handleChangeBrushSize = (v: number) => {
    dispatch(setShouldShowBrushPreview(true));
    dispatch(setBrushSize(v));
  };

  // Hotkeys

  // Decrease brush size
  useHotkeys(
    '[',
    (e: KeyboardEvent) => {
      e.preventDefault();
      if (brushSize - 5 > 0) {
        handleChangeBrushSize(brushSize - 5);
      } else {
        handleChangeBrushSize(1);
      }
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowMask, brushSize]
  );

  // Increase brush size
  useHotkeys(
    ']',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeBrushSize(brushSize + 5);
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowMask, brushSize]
  );

  // Set tool to brush
  useHotkeys(
    'b',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleSelectBrushTool();
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowMask]
  );

  return (
    <IAIPopover
      trigger="hover"
      onOpen={handleShowBrushPreview}
      onClose={handleHideBrushPreview}
      triggerComponent={
        <IAIIconButton
          aria-label="Brush (B)"
          tooltip="Brush (B)"
          icon={<FaPaintBrush />}
          onClick={handleSelectBrushTool}
          data-selected={tool === 'brush'}
          isDisabled={!shouldShowMask}
        />
      }
    >
      <div className="inpainting-brush-options">
        <IAISlider
          label="Brush Size"
          value={brushSize}
          onChange={handleChangeBrushSize}
          min={1}
          max={200}
        />
        <IAINumberInput
          value={brushSize}
          onChange={handleChangeBrushSize}
          width={'80px'}
          min={1}
          max={999}
          isDisabled={!shouldShowMask}
        />
        <IAICanvasMaskColorPicker />
      </div>
    </IAIPopover>
  );
}
