import IAICanvasBrushControl from './IAICanvasControls/IAICanvasBrushControl';
import IAICanvasEraserControl from './IAICanvasControls/IAICanvasEraserControl';
import IAICanvasUndoControl from './IAICanvasControls/IAICanvasUndoButton';
import IAICanvasRedoControl from './IAICanvasControls/IAICanvasRedoButton';
import { ButtonGroup } from '@chakra-ui/react';
import IAICanvasMaskClear from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskClear';
import IAICanvasMaskVisibilityControl from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskVisibilityControl';
import IAICanvasMaskInvertControl from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskInvertControl';
import IAICanvasLockBoundingBoxControl from './IAICanvasControls/IAICanvasLockBoundingBoxControl';
import IAICanvasShowHideBoundingBoxControl from './IAICanvasControls/IAICanvasShowHideBoundingBoxControl';
import ImageUploaderIconButton from 'common/components/ImageUploaderIconButton';
import { createSelector } from '@reduxjs/toolkit';
import {
  outpaintingCanvasSelector,
  OutpaintingCanvasState,
} from './canvasSlice';
import { RootState } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { OptionsState } from 'features/options/optionsSlice';
import _ from 'lodash';

export const canvasControlsSelector = createSelector(
  [
    outpaintingCanvasSelector,
    (state: RootState) => state.options,
    activeTabNameSelector,
  ],
  (
    outpaintingCanvas: OutpaintingCanvasState,
    options: OptionsState,
    activeTabName
  ) => {
    const { stageScale, boundingBoxCoordinates, boundingBoxDimensions } =
      outpaintingCanvas;
    return {
      activeTabName,
      stageScale,
      boundingBoxCoordinates,
      boundingBoxDimensions,
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
