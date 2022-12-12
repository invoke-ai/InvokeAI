import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { mergeAndUploadCanvas } from 'features/canvas/store/thunks/mergeAndUploadCanvas';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaCopy } from 'react-icons/fa';

export default function UnifiedCanvasCopyToClipboard() {
  const isStaging = useAppSelector(isStagingSelector);
  const canvasBaseLayer = getCanvasBaseLayer();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const shouldCropToBoundingBoxOnSave = useAppSelector(
    (state: RootState) => state.canvas.shouldCropToBoundingBoxOnSave
  );

  const dispatch = useAppDispatch();

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

  const handleCopyImageToClipboard = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
        cropToBoundingBox: shouldCropToBoundingBoxOnSave,
        shouldCopy: true,
      })
    );
  };

  return (
    <IAIIconButton
      aria-label="Copy to Clipboard (Cmd/Ctrl+C)"
      tooltip="Copy to Clipboard (Cmd/Ctrl+C)"
      icon={<FaCopy />}
      onClick={handleCopyImageToClipboard}
      isDisabled={isStaging}
    />
  );
}
