import { ButtonGroup } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  resizeAndScaleCanvas,
  resetCanvas,
  resetCanvasView,
  setTool,
  fitBoundingBoxToStage,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import { canvasImageLayerRef, stageRef } from '../IAICanvas';
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
import IAICanvasUndoButton from './IAICanvasUndoButton';
import IAICanvasRedoButton from './IAICanvasRedoButton';
import IAICanvasSettingsButtonPopover from './IAICanvasSettingsButtonPopover';
import IAICanvasEraserButtonPopover from './IAICanvasEraserButtonPopover';
import IAICanvasBrushButtonPopover from './IAICanvasBrushButtonPopover';
import IAICanvasMaskButtonPopover from './IAICanvasMaskButtonPopover';
import { mergeAndUploadCanvas } from 'features/canvas/util/mergeAndUploadCanvas';
import IAICheckbox from 'common/components/IAICheckbox';
import { ChangeEvent } from 'react';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';

export const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { tool } = canvas;
    return {
      tool,
      isStaging,
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
  const { tool, isStaging } = useAppSelector(selector);

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
              mergeAndUploadCanvas({
                canvasImageLayerRef,
                cropVisible: true,
                saveToGallery: true,
              })
            );
          }}
        />
        <IAIIconButton
          aria-label="Copy Selection"
          tooltip="Copy Selection"
          icon={<FaCopy />}
          onClick={() => {
            dispatch(
              mergeAndUploadCanvas({
                canvasImageLayerRef,
                cropVisible: true,
                copyAfterSaving: true,
              })
            );
          }}
        />
        <IAIIconButton
          aria-label="Download Selection"
          tooltip="Download Selection"
          icon={<FaDownload />}
          onClick={() => {
            dispatch(
              mergeAndUploadCanvas({
                canvasImageLayerRef,
                cropVisible: true,
                downloadAfterSaving: true,
              })
            );
          }}
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
    </div>
  );
};

export default IAICanvasOutpaintingControls;
