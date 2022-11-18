import { ButtonGroup } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  resetCanvas,
  resetCanvasView,
  resizeAndScaleCanvas,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
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
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  getCanvasBaseLayer,
  getCanvasStage,
} from 'features/canvas/util/konvaInstanceProvider';

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
  const canvasBaseLayer = getCanvasBaseLayer();

  useHotkeys(
    ['m'],
    () => {
      handleSelectMoveTool();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    []
  );

  useHotkeys(
    ['r'],
    () => {
      handleResetCanvasView();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  useHotkeys(
    ['shift+m'],
    () => {
      handleMergeVisible();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  useHotkeys(
    ['shift+s'],
    () => {
      handleSaveToGallery();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  useHotkeys(
    ['meta+c', 'ctrl+c'],
    () => {
      handleCopyImageToClipboard();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  useHotkeys(
    ['shift+d'],
    () => {
      handleDownloadAsImage();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  const handleSelectMoveTool = () => dispatch(setTool('move'));

  const handleResetCanvasView = () => {
    const canvasBaseLayer = getCanvasBaseLayer();
    if (!canvasBaseLayer) return;
    const clientRect = canvasBaseLayer.getClientRect({
      skipTransform: true,
    });
    dispatch(
      resetCanvasView({
        contentRect: clientRect,
      })
    );
  };

  const handleResetCanvas = () => {
    dispatch(resetCanvas());
    dispatch(resizeAndScaleCanvas());
  };

  const handleMergeVisible = () => {
    dispatch(mergeAndUploadCanvas({}));
  };

  const handleSaveToGallery = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: true,
        saveToGallery: true,
      })
    );
  };

  const handleCopyImageToClipboard = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: true,
        copyAfterSaving: true,
      })
    );
  };

  const handleDownloadAsImage = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: true,
        downloadAfterSaving: true,
      })
    );
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
          onClick={handleSelectMoveTool}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label="Merge Visible (Shift + M)"
          tooltip="Merge Visible (Shift + M)"
          icon={<FaLayerGroup />}
          onClick={handleMergeVisible}
        />
        <IAIIconButton
          aria-label="Save to Gallery (Shift + S)"
          tooltip="Save to Gallery (Shift + S)"
          icon={<FaSave />}
          onClick={handleSaveToGallery}
        />
        <IAIIconButton
          aria-label="Copy to Clipboard (Cmd/Ctrl + C)"
          tooltip="Copy to Clipboard (Cmd/Ctrl + C)"
          icon={<FaCopy />}
          onClick={handleCopyImageToClipboard}
        />
        <IAIIconButton
          aria-label="Download as Image (Shift + D)"
          tooltip="Download as Image (Shift + D)"
          icon={<FaDownload />}
          onClick={handleDownloadAsImage}
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
          aria-label="Reset Canvas View (R)"
          tooltip="Reset Canvas View (R)"
          icon={<FaCrosshairs />}
          onClick={handleResetCanvasView}
        />
        <IAIIconButton
          aria-label="Reset Canvas"
          tooltip="Reset Canvas"
          icon={<FaTrash />}
          onClick={handleResetCanvas}
        />
      </ButtonGroup>
    </div>
  );
};

export default IAICanvasOutpaintingControls;
