import { Box, ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useSingleAndDoubleClick } from 'common/hooks/useSingleAndDoubleClick';
import {
  canvasCopiedToClipboard,
  canvasDownloadedAsImage,
  canvasMerged,
  canvasSavedToGallery,
} from 'features/canvas/store/actions';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  resetCanvas,
  resetCanvasView,
  setIsMaskEnabled,
  setLayer,
  setTool,
} from 'features/canvas/store/canvasSlice';
import {
  CanvasLayer,
  LAYER_NAMES_DICT,
} from 'features/canvas/store/canvasTypes';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { systemSelector } from 'features/system/store/systemSelectors';
import { useCopyImageToClipboard } from 'features/ui/hooks/useCopyImageToClipboard';
import { isEqual } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
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
import IAICanvasMaskOptions from './IAICanvasMaskOptions';
import IAICanvasRedoButton from './IAICanvasRedoButton';
import IAICanvasSettingsButtonPopover from './IAICanvasSettingsButtonPopover';
import IAICanvasToolChooserOptions from './IAICanvasToolChooserOptions';
import IAICanvasUndoButton from './IAICanvasUndoButton';
import { memo } from 'react';

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
      resultEqualityCheck: isEqual,
    },
  }
);

const IAICanvasToolbar = () => {
  const dispatch = useAppDispatch();
  const { isProcessing, isStaging, isMaskEnabled, layer, tool } =
    useAppSelector(selector);
  const canvasBaseLayer = getCanvasBaseLayer();

  const { t } = useTranslation();
  const { isClipboardAPIAvailable } = useCopyImageToClipboard();

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction: { type: 'SET_CANVAS_INITIAL_IMAGE' },
  });

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
      enabled: () => !isStaging && isClipboardAPIAvailable,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing, isClipboardAPIAvailable]
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

  const handleClickResetCanvasView = useSingleAndDoubleClick(
    () => handleResetCanvasView(false),
    () => handleResetCanvasView(true)
  );

  const handleResetCanvasView = (shouldScaleTo1 = false) => {
    const canvasBaseLayer = getCanvasBaseLayer();
    if (!canvasBaseLayer) {
      return;
    }
    const clientRect = canvasBaseLayer.getClientRect({
      skipTransform: true,
    });
    dispatch(
      resetCanvasView({
        contentRect: clientRect,
        shouldScaleTo1,
      })
    );
  };

  const handleResetCanvas = () => {
    dispatch(resetCanvas());
  };

  const handleMergeVisible = () => {
    dispatch(canvasMerged());
  };

  const handleSaveToGallery = () => {
    dispatch(canvasSavedToGallery());
  };

  const handleCopyImageToClipboard = () => {
    if (!isClipboardAPIAvailable) {
      return;
    }
    dispatch(canvasCopiedToClipboard());
  };

  const handleDownloadAsImage = () => {
    dispatch(canvasDownloadedAsImage());
  };

  const handleChangeLayer = (v: string) => {
    const newLayer = v as CanvasLayer;
    dispatch(setLayer(newLayer));
    if (newLayer === 'mask' && !isMaskEnabled) {
      dispatch(setIsMaskEnabled(true));
    }
  };

  return (
    <Flex
      sx={{
        alignItems: 'center',
        gap: 2,
        flexWrap: 'wrap',
      }}
    >
      <Box w={24}>
        <IAIMantineSelect
          tooltip={`${t('unifiedCanvas.layer')} (Q)`}
          value={layer}
          data={LAYER_NAMES_DICT}
          onChange={handleChangeLayer}
          disabled={isStaging}
        />
      </Box>

      <IAICanvasMaskOptions />
      <IAICanvasToolChooserOptions />

      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label={`${t('unifiedCanvas.move')} (V)`}
          tooltip={`${t('unifiedCanvas.move')} (V)`}
          icon={<FaArrowsAlt />}
          isChecked={tool === 'move' || isStaging}
          onClick={handleSelectMoveTool}
        />
        <IAIIconButton
          aria-label={`${t('unifiedCanvas.resetView')} (R)`}
          tooltip={`${t('unifiedCanvas.resetView')} (R)`}
          icon={<FaCrosshairs />}
          onClick={handleClickResetCanvasView}
        />
      </ButtonGroup>

      <ButtonGroup isAttached>
        <IAIIconButton
          aria-label={`${t('unifiedCanvas.mergeVisible')} (Shift+M)`}
          tooltip={`${t('unifiedCanvas.mergeVisible')} (Shift+M)`}
          icon={<FaLayerGroup />}
          onClick={handleMergeVisible}
          isDisabled={isStaging}
        />
        <IAIIconButton
          aria-label={`${t('unifiedCanvas.saveToGallery')} (Shift+S)`}
          tooltip={`${t('unifiedCanvas.saveToGallery')} (Shift+S)`}
          icon={<FaSave />}
          onClick={handleSaveToGallery}
          isDisabled={isStaging}
        />
        {isClipboardAPIAvailable && (
          <IAIIconButton
            aria-label={`${t('unifiedCanvas.copyToClipboard')} (Cmd/Ctrl+C)`}
            tooltip={`${t('unifiedCanvas.copyToClipboard')} (Cmd/Ctrl+C)`}
            icon={<FaCopy />}
            onClick={handleCopyImageToClipboard}
            isDisabled={isStaging}
          />
        )}
        <IAIIconButton
          aria-label={`${t('unifiedCanvas.downloadAsImage')} (Shift+D)`}
          tooltip={`${t('unifiedCanvas.downloadAsImage')} (Shift+D)`}
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
          aria-label={`${t('common.upload')}`}
          tooltip={`${t('common.upload')}`}
          icon={<FaUpload />}
          isDisabled={isStaging}
          {...getUploadButtonProps()}
        />
        <input {...getUploadInputProps()} />
        <IAIIconButton
          aria-label={`${t('unifiedCanvas.clearCanvas')}`}
          tooltip={`${t('unifiedCanvas.clearCanvas')}`}
          icon={<FaTrash />}
          onClick={handleResetCanvas}
          colorScheme="error"
          isDisabled={isStaging}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <IAICanvasSettingsButtonPopover />
      </ButtonGroup>
    </Flex>
  );
};

export default memo(IAICanvasToolbar);
