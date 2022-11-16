import { ButtonGroup } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  currentCanvasSelector,
  resizeAndScaleCanvas,
  isStagingSelector,
  resetCanvas,
  resetCanvasView,
  setShouldLockToInitialImage,
  setTool,
} from './canvasSlice';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import { canvasImageLayerRef, stageRef } from './IAICanvas';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  FaArrowsAlt,
  FaCopy,
  FaCrosshairs,
  FaDownload,
  FaLayerGroup,
  FaSave,
  FaTrash,
  FaUpload,
} from 'react-icons/fa';
import IAICanvasUndoButton from './IAICanvasControls/IAICanvasUndoButton';
import IAICanvasRedoButton from './IAICanvasControls/IAICanvasRedoButton';
import IAICanvasSettingsButtonPopover from './IAICanvasSettingsButtonPopover';
import IAICanvasEraserButtonPopover from './IAICanvasEraserButtonPopover';
import IAICanvasBrushButtonPopover from './IAICanvasBrushButtonPopover';
import IAICanvasMaskButtonPopover from './IAICanvasMaskButtonPopover';
import { mergeAndUploadCanvas } from './util/mergeAndUploadCanvas';
import IAICheckbox from 'common/components/IAICheckbox';
import { ChangeEvent } from 'react';

export const canvasControlsSelector = createSelector(
  [
    (state: RootState) => state.canvas,
    currentCanvasSelector,
    isStagingSelector,
  ],
  (canvas, currentCanvas, isStaging) => {
    const { shouldLockToInitialImage } = canvas;
    const { tool } = currentCanvas;
    return {
      tool,
      isStaging,
      shouldLockToInitialImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const IAICanvasOutpaintingControls = () => {
  const dispatch = useAppDispatch();
  const { tool, isStaging, shouldLockToInitialImage } = useAppSelector(
    canvasControlsSelector
  );

  const handleToggleShouldLockToInitialImage = (
    e: ChangeEvent<HTMLInputElement>
  ) => {
    dispatch(setShouldLockToInitialImage(e.target.checked));
    dispatch(resizeAndScaleCanvas());
  };

  return (
    <div className="inpainting-settings">
      <IAICanvasMaskButtonPopover />
      <ButtonGroup isAttached>
        <IAICanvasBrushButtonPopover />
        <IAICanvasEraserButtonPopover />
        <IAIIconButton
          aria-label="Move (M)"
          tooltip="Move (M)"
          icon={<FaArrowsAlt />}
          data-selected={tool === 'move' || isStaging}
          onClick={() => dispatch(setTool('move'))}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label="Merge Visible"
          tooltip="Merge Visible"
          icon={<FaLayerGroup />}
          onClick={() => {
            dispatch(
              mergeAndUploadCanvas({
                canvasImageLayerRef,
                saveToGallery: false,
              })
            );
          }}
        />
        <IAIIconButton
          aria-label="Save to Gallery"
          tooltip="Save to Gallery"
          icon={<FaSave />}
          onClick={() => {
            dispatch(
              mergeAndUploadCanvas({ canvasImageLayerRef, saveToGallery: true })
            );
          }}
        />
        <IAIIconButton
          aria-label="Copy Selection"
          tooltip="Copy Selection"
          icon={<FaCopy />}
        />
        <IAIIconButton
          aria-label="Download Selection"
          tooltip="Download Selection"
          icon={<FaDownload />}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAICanvasUndoButton />
        <IAICanvasRedoButton />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAICanvasSettingsButtonPopover />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label="Upload"
          tooltip="Upload"
          icon={<FaUpload />}
        />
        <IAIIconButton
          aria-label="Reset Canvas View"
          tooltip="Reset Canvas View"
          icon={<FaCrosshairs />}
          onClick={() => {
            if (!stageRef.current || !canvasImageLayerRef.current) return;
            const clientRect = canvasImageLayerRef.current.getClientRect({
              skipTransform: true,
            });
            dispatch(
              resetCanvasView({
                contentRect: clientRect,
              })
            );
          }}
        />
        <IAIIconButton
          aria-label="Reset Canvas"
          tooltip="Reset Canvas"
          icon={<FaTrash />}
          onClick={() => dispatch(resetCanvas())}
        />
      </ButtonGroup>
      <IAICheckbox
        label={'Lock Canvas to Initial Image'}
        isChecked={shouldLockToInitialImage}
        onChange={handleToggleShouldLockToInitialImage}
      />
    </div>
  );
};

export default IAICanvasOutpaintingControls;
