import { ButtonGroup } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  currentCanvasSelector,
  isStagingSelector,
  resetCanvas,
  setTool,
  uploadOutpaintingMergedImage,
} from './canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import { canvasImageLayerRef } from './IAICanvas';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  FaArrowsAlt,
  FaCopy,
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

export const canvasControlsSelector = createSelector(
  [currentCanvasSelector, isStagingSelector],
  (currentCanvas, isStaging) => {
    const { tool } = currentCanvas;

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
  const { tool, isStaging } = useAppSelector(canvasControlsSelector);

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
            dispatch(uploadOutpaintingMergedImage(canvasImageLayerRef));
          }}
        />
        <IAIIconButton
          aria-label="Save Selection to Gallery"
          tooltip="Save Selection to Gallery"
          icon={<FaSave />}
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
