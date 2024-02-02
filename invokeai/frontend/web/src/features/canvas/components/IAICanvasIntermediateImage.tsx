import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { selectProgressSlice } from 'features/progress/store/progressSlice';
import { memo, useEffect, useState } from 'react';
import { Image as KonvaImage } from 'react-konva';

const progressImageSelector = createMemoizedSelector([selectProgressSlice, selectCanvasSlice], (progress, canvas) => {
  return {
    progressImage: progress.canvasDenoiseProgress?.progress_image,
    boundingBox: canvas.layerState.stagingArea.boundingBox,
  };
});

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
