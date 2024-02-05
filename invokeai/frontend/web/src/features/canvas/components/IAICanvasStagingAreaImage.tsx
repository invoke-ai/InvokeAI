import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { $authToken } from 'app/store/nanostores/authToken';
import { useAppSelector } from 'app/store/storeHooks';
import IAICanvasIntermediateImage, {
  progressImageSelector,
} from 'features/canvas/components/IAICanvasIntermediateImage';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { selectProgressSlice } from 'features/progress/store/progressSlice';
import { memo } from 'react';
import { Image } from 'react-konva';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import useImage from 'use-image';

import IAICanvasImageErrorFallback from './IAICanvasImageErrorFallback';

const selector = createMemoizedSelector([selectProgressSlice, selectCanvasSlice], (progress, canvas) => {
  const { selectedImageIndex, images } = canvas.layerState.stagingArea;

  const currentStagingAreaImage =
    images.length > 0 && selectedImageIndex !== undefined ? images[selectedImageIndex] : undefined;

  const progressImage =
    progress.latestDenoiseProgress && progress.canvasBatchIds.includes(progress.latestDenoiseProgress.queue_batch_id)
      ? progress.latestDenoiseProgress.progress_image
      : undefined;

  const boundingBox = canvas.layerState.stagingArea.boundingBox ?? {
    ...canvas.boundingBoxCoordinates,
    ...canvas.boundingBoxDimensions,
  };

  return {
    currentStagingAreaImage,
    progressImage,
    boundingBox,
  };
});

export const IAICanvasStagingAreaImage = memo(() => {
  const { currentStagingAreaImage, boundingBox } = useAppSelector(selector);
  const { progressImage } = useAppSelector(progressImageSelector);
  const { currentData: imageDTO, isError } = useGetImageDTOQuery(currentStagingAreaImage?.imageName ?? skipToken);
  const [stagedImageEl, stagedImageElStatus] = useImage(
    imageDTO?.image_url ?? '',
    $authToken.get() ? 'use-credentials' : 'anonymous'
  );

  if (currentStagingAreaImage && (isError || stagedImageElStatus === 'failed')) {
    return <IAICanvasImageErrorFallback canvasImage={currentStagingAreaImage} />;
  }

  if (progressImage) {
    return <IAICanvasIntermediateImage />;
  }

  if (stagedImageEl) {
    return (
      <Image
        x={boundingBox.x}
        y={boundingBox.y}
        width={boundingBox.width}
        height={boundingBox.height}
        image={stagedImageEl}
        listening={false}
      />
    );
  }
});

IAICanvasStagingAreaImage.displayName = 'IAICanvasStagingAreaImage';
