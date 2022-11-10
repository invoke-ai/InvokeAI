import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaPaintBrush } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import IAINumberInput from 'common/components/IAINumberInput';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

import {
  currentCanvasSelector,
  setBrushSize,
  setShouldShowBrushPreview,
  setTool,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import IAICanvasMaskColorPicker from './IAICanvasMaskControls/IAICanvasMaskColorPicker';

const inpaintingBrushSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (currentCanvas, activeTabName) => {
    const { tool, brushSize } = currentCanvas;

    return {
      tool,
      brushSize,
      activeTabName,
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
  const { tool, brushSize, activeTabName } = useAppSelector(
    inpaintingBrushSelector
  );

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
      enabled: true,
    },
    [activeTabName, brushSize]
  );

  // Increase brush size
  useHotkeys(
    ']',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeBrushSize(brushSize + 5);
    },
    {
      enabled: true,
    },
    [activeTabName, brushSize]
  );

  // Set tool to brush
  useHotkeys(
    'b',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleSelectBrushTool();
    },
    {
      enabled: true,
    },
    [activeTabName]
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
        />
        <IAICanvasMaskColorPicker />
      </div>
    </IAIPopover>
  );
}
