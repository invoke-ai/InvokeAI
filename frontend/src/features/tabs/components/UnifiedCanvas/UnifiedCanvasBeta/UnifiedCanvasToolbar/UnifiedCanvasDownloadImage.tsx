import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { mergeAndUploadCanvas } from 'features/canvas/store/thunks/mergeAndUploadCanvas';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaDownload } from 'react-icons/fa';

export default function UnifiedCanvasDownloadImage() {
  const dispatch = useAppDispatch();
  const canvasBaseLayer = getCanvasBaseLayer();

  const isStaging = useAppSelector(isStagingSelector);

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const shouldCropToBoundingBoxOnSave = useAppSelector(
    (state: RootState) => state.canvas.shouldCropToBoundingBoxOnSave
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

  const handleDownloadAsImage = () => {
    dispatch(
      mergeAndUploadCanvas({
        cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
        cropToBoundingBox: shouldCropToBoundingBoxOnSave,
        shouldDownload: true,
      })
    );
  };
  return (
    <IAIIconButton
      aria-label="Download as Image (Shift+D)"
      tooltip="Download as Image (Shift+D)"
      icon={<FaDownload />}
      onClick={handleDownloadAsImage}
      isDisabled={isStaging}
    />
  );
}
