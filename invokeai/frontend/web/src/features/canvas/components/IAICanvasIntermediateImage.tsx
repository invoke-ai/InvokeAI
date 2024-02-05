import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { selectProgressSlice } from 'features/progress/store/progressSlice';
import { memo, useEffect, useState } from 'react';
import { Image as KonvaImage } from 'react-konva';

export const progressImageSelector = createMemoizedSelector(
  [selectProgressSlice, selectCanvasSlice],
  (progress, canvas) => {
    const isLatestProgressFromCanvas =
      progress.latestDenoiseProgress && progress.canvasBatchIds.includes(progress.latestDenoiseProgress.queue_batch_id);

    const { selectedImageIndex, images } = canvas.layerState.stagingArea;
    const _currentStagingAreaImage =
      images.length > 0 && selectedImageIndex !== undefined ? images[selectedImageIndex] : undefined;

    const isProgressImageIncomplete =
      progress.latestDenoiseProgress?.graph_execution_state_id !==
      progress.latestImageOutputEvent?.graph_execution_state_id;

    return {
      progressImage:
        progress.latestDenoiseProgress && isLatestProgressFromCanvas && isProgressImageIncomplete
          ? progress.latestDenoiseProgress.progress_image
          : undefined,
      boundingBox: canvas.layerState.stagingArea.boundingBox,
    };
  }
);

const IAICanvasIntermediateImage = () => {
  const { progressImage, boundingBox } = useAppSelector(progressImageSelector);
  const [loadedImageElement, setLoadedImageElement] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!progressImage) {
      return;
    }

    const tempImage = new Image();

    tempImage.onload = () => {
      setLoadedImageElement(tempImage);
    };

    tempImage.src = progressImage.dataURL;
  }, [progressImage]);

  if (!(progressImage && boundingBox) || !loadedImageElement) {
    return null;
  }

  return (
    <KonvaImage
      x={boundingBox.x}
      y={boundingBox.y}
      width={boundingBox.width}
      height={boundingBox.height}
      image={loadedImageElement}
      listening={false}
    />
  );
};

export default memo(IAICanvasIntermediateImage);
