import IAICanvasBrushControl from './IAICanvasControls/IAICanvasBrushControl';
import IAICanvasEraserControl from './IAICanvasControls/IAICanvasEraserControl';
import IAICanvasUndoControl from './IAICanvasControls/IAICanvasUndoButton';
import IAICanvasRedoControl from './IAICanvasControls/IAICanvasRedoButton';
import { Button, ButtonGroup } from '@chakra-ui/react';
import IAICanvasMaskClear from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskClear';
import IAICanvasMaskVisibilityControl from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskVisibilityControl';
import IAICanvasMaskInvertControl from './IAICanvasControls/IAICanvasMaskControls/IAICanvasMaskInvertControl';
import IAICanvasLockBoundingBoxControl from './IAICanvasControls/IAICanvasLockBoundingBoxControl';
import IAICanvasShowHideBoundingBoxControl from './IAICanvasControls/IAICanvasShowHideBoundingBoxControl';
import ImageUploaderIconButton from 'common/components/ImageUploaderIconButton';
import { createSelector } from '@reduxjs/toolkit';
import {
  currentCanvasSelector,
  GenericCanvasState,
  outpaintingCanvasSelector,
  OutpaintingCanvasState,
  uploadOutpaintingMergedImage,
} from './canvasSlice';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { OptionsState } from 'features/options/optionsSlice';
import _ from 'lodash';
import IAICanvasImageEraserControl from './IAICanvasControls/IAICanvasImageEraserControl';
import { canvasImageLayerRef } from './IAICanvas';
import { uploadImage } from 'app/socketio/actions';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaSave } from 'react-icons/fa';

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
  const dispatch = useAppDispatch();
  const {
    activeTabName,
    boundingBoxCoordinates,
    boundingBoxDimensions,
    stageScale,
  } = useAppSelector(canvasControlsSelector);

  return (
    <div className="inpainting-settings">
      <ButtonGroup isAttached={true}>
        <IAICanvasBrushControl />
        <IAICanvasEraserControl />
        {activeTabName === 'outpainting' && <IAICanvasImageEraserControl />}
      </ButtonGroup>
      <ButtonGroup isAttached={true}>
        <IAICanvasMaskVisibilityControl />
        <IAICanvasMaskInvertControl />
        <IAICanvasLockBoundingBoxControl />
        <IAICanvasShowHideBoundingBoxControl />
        <IAICanvasMaskClear />
      </ButtonGroup>
      <IAIIconButton
        aria-label="Save"
        tooltip="Save"
        icon={<FaSave />}
        onClick={() => {
          dispatch(uploadOutpaintingMergedImage(canvasImageLayerRef));
        }}
        fontSize={20}
      />
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
