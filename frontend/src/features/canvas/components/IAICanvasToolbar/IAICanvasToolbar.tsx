import { ButtonGroup } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  resetCanvas,
  resetCanvasView,
  resizeAndScaleCanvas,
  setIsMaskEnabled,
  setLayer,
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
import IAICanvasMaskOptions from './IAICanvasMaskOptions';
import { mergeAndUploadCanvas } from 'features/canvas/store/thunks/mergeAndUploadCanvas';
import { useHotkeys } from 'react-hotkeys-hook';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { systemSelector } from 'features/system/store/systemSelectors';
import IAICanvasToolChooserOptions from './IAICanvasToolChooserOptions';
import useImageUploader from 'common/hooks/useImageUploader';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import IAISelect from 'common/components/IAISelect';
import {
  CanvasLayer,
  LAYER_NAMES_DICT,
} from 'features/canvas/store/canvasTypes';
import { ChangeEvent } from 'react';

export const selector = createSelector(
  [systemSelector, canvasSelector, isStagingSelector],
  (system, canvas, isStaging) => {
    const { isProcessing } = system;
    const { tool, shouldCropToBoundingBoxOnSave, layer, isMaskEnabled } =
      canvas;

    return {
      isProcessing,
      isStaging,
      isMaskEnabled,
      tool,
      layer,
      shouldCropToBoundingBoxOnSave,
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
  const {
    isProcessing,
    isStaging,
    isMaskEnabled,
    layer,
    tool,
    shouldCropToBoundingBoxOnSave,
  } = useAppSelector(selector);
  const canvasBaseLayer = getCanvasBaseLayer();

  const { openUploader } = useImageUploader();

  useHotkeys(
    ['v'],
    () => {
      handleSelectMoveTool();
    },
    {
      enabled: () => !isStaging,
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
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  useHotkeys(
    ['shift+s'],
    () => {
      handleSaveToGallery();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  useHotkeys(
    ['meta+c', 'ctrl+c'],
    () => {
      handleCopyImageToClipboard();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  useHotkeys(
    ['shift+d'],
    () => {
      handleDownloadAsImage();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
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
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: false,
        shouldSetAsInitialImage: true,
      })
    );
  };

  const handleSaveToGallery = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
        cropToBoundingBox: shouldCropToBoundingBoxOnSave,
        shouldSaveToGallery: true,
      })
    );
  };

  const handleCopyImageToClipboard = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
        cropToBoundingBox: shouldCropToBoundingBoxOnSave,
        shouldCopy: true,
      })
    );
  };

  const handleDownloadAsImage = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
        cropToBoundingBox: shouldCropToBoundingBoxOnSave,
        shouldDownload: true,
      })
    );
  };

  const handleChangeLayer = (e: ChangeEvent<HTMLSelectElement>) => {
    const newLayer = e.target.value as CanvasLayer;
    dispatch(setLayer(newLayer));
    if (newLayer === 'mask' && !isMaskEnabled) {
      dispatch(setIsMaskEnabled(true));
    }
  };

  return (
    <div className="inpainting-settings">
      <IAISelect
        tooltip={'Layer (Q)'}
        tooltipProps={{ hasArrow: true, placement: 'top' }}
        value={layer}
        validValues={LAYER_NAMES_DICT}
        onChange={handleChangeLayer}
        isDisabled={isStaging}
      />

      <IAICanvasMaskOptions />
      <IAICanvasToolChooserOptions />

      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label="Move Tool (V)"
          tooltip="Move Tool (V)"
          icon={<FaArrowsAlt />}
          data-selected={tool === 'move' || isStaging}
          onClick={handleSelectMoveTool}
        />
        <IAIIconButton
          aria-label="Reset View (R)"
          tooltip="Reset View (R)"
          icon={<FaCrosshairs />}
          onClick={handleResetCanvasView}
        />
      </ButtonGroup>

      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label="Merge Visible (Shift+M)"
          tooltip="Merge Visible (Shift+M)"
          icon={<FaLayerGroup />}
          onClick={handleMergeVisible}
          isDisabled={isStaging}
        />
        <IAIIconButton
          aria-label="Save to Gallery (Shift+S)"
          tooltip="Save to Gallery (Shift+S)"
          icon={<FaSave />}
          onClick={handleSaveToGallery}
          isDisabled={isStaging}
        />
        <IAIIconButton
          aria-label="Copy to Clipboard (Cmd/Ctrl+C)"
          tooltip="Copy to Clipboard (Cmd/Ctrl+C)"
          icon={<FaCopy />}
          onClick={handleCopyImageToClipboard}
          isDisabled={isStaging}
        />
        <IAIIconButton
          aria-label="Download as Image (Shift+D)"
          tooltip="Download as Image (Shift+D)"
          icon={<FaDownload />}
          onClick={handleDownloadAsImage}
          isDisabled={isStaging}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAICanvasUndoButton />
        <IAICanvasRedoButton />
      </ButtonGroup>

      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label="Upload"
          tooltip="Upload"
          icon={<FaUpload />}
          onClick={openUploader}
          isDisabled={isStaging}
        />
        <IAIIconButton
          aria-label="Clear Canvas"
          tooltip="Clear Canvas"
          icon={<FaTrash />}
          onClick={handleResetCanvas}
          style={{ backgroundColor: 'var(--btn-delete-image)' }}
          isDisabled={isStaging}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAICanvasSettingsButtonPopover />
      </ButtonGroup>
    </div>
  );
};

export default IAICanvasOutpaintingControls;
