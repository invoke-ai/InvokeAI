import IAICanvasBrushControl from './IAICanvasControls/IAICanvasBrushControl';
import IAICanvasEraserControl from './IAICanvasControls/IAICanvasEraserControl';
import IAICanvasUndoControl from './IAICanvasControls/IAICanvasUndoControl';
import IAICanvasRedoControl from './IAICanvasControls/IAICanvasRedoControl';
import { ButtonGroup } from '@chakra-ui/react';
import IAICanvasMaskClear from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskClear';
import IAICanvasMaskVisibilityControl from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskVisibilityControl';
import IAICanvasMaskInvertControl from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskInvertControl';
import IAICanvasLockBoundingBoxControl from './IAICanvasControls/IAICanvasLockBoundingBoxControl';
import IAICanvasShowHideBoundingBoxControl from './IAICanvasControls/IAICanvasShowHideBoundingBoxControl';
import ImageUploaderIconButton from 'common/components/ImageUploaderIconButton';
import { createSelector } from '@reduxjs/toolkit';
import { currentCanvasSelector, GenericCanvasState } from './canvasSlice';
import { RootState } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { OptionsState } from 'features/options/optionsSlice';
import _ from 'lodash';


export const canvasControlsSelector = createSelector(
  [
    currentCanvasSelector,
    (state: RootState) => state.options,
    activeTabNameSelector,
  ],
  (currentCanvas: GenericCanvasState, options: OptionsState, activeTabName) => {
    const {
      tool,
      brushSize,
      maskColor,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      lines,
      pastLines,
      futureLines,
      shouldShowBoundingBoxFill,
    } = currentCanvas;

    const { showDualDisplay } = options;

    return {
      tool,
      brushSize,
      maskColor,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      canUndo: pastLines.length > 0,
      canRedo: futureLines.length > 0,
      isMaskEmpty: lines.length === 0,
      activeTabName,
      showDualDisplay,
      shouldShowBoundingBoxFill,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const IAICanvasControls = () => {
  return (
    <div className="inpainting-settings">
      <ButtonGroup isAttached={true}>
        <IAICanvasBrushControl />
        <IAICanvasEraserControl />
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAICanvasMaskVisibilityControl />
        <IAICanvasMaskInvertControl />
        <IAICanvasLockBoundingBoxControl />
        <IAICanvasShowHideBoundingBoxControl />
        <IAICanvasMaskClear />
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAICanvasUndoControl />
        <IAICanvasRedoControl />
      </ButtonGroup>
      <ButtonGroup isAttached={true}>
        <ImageUploaderIconButton />
      </ButtonGroup>
    </div>
  );
};

export default IAICanvasControls;
