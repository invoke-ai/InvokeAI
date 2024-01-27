import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import type { GroupConfig } from 'konva/lib/Group';
import { memo } from 'react';
import { Group, Rect } from 'react-konva';

import IAICanvasImage from './IAICanvasImage';

const dash = [4, 4];

const selector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  const {
    layerState,
    shouldShowStagingImage,
    shouldShowStagingOutline,
    boundingBoxCoordinates: stageBoundingBoxCoordinates,
    boundingBoxDimensions: stageBoundingBoxDimensions,
  } = canvas;

  const { selectedImageIndex, images, boundingBox } = layerState.stagingArea;

  return {
    currentStagingAreaImage:
      images.length > 0 && selectedImageIndex !== undefined ? images[selectedImageIndex] : undefined,
    isOnFirstImage: selectedImageIndex === 0,
    isOnLastImage: selectedImageIndex === images.length - 1,
    shouldShowStagingImage,
    shouldShowStagingOutline,
    x: boundingBox?.x ?? stageBoundingBoxCoordinates.x,
    y: boundingBox?.y ?? stageBoundingBoxCoordinates.y,
    width: boundingBox?.width ?? stageBoundingBoxDimensions.width,
    height: boundingBox?.height ?? stageBoundingBoxDimensions.height,
  };
});

type Props = GroupConfig;

const IAICanvasStagingArea = (props: Props) => {
  const { currentStagingAreaImage, shouldShowStagingImage, shouldShowStagingOutline, x, y, width, height } =
    useAppSelector(selector);

  return (
    <Group {...props}>
      {shouldShowStagingImage && currentStagingAreaImage && <IAICanvasImage canvasImage={currentStagingAreaImage} />}
      {shouldShowStagingOutline && (
        <Group listening={false}>
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            strokeWidth={1}
            stroke="white"
            strokeScaleEnabled={false}
            listening={false}
          />
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            dash={dash}
            strokeWidth={1}
            stroke="black"
            strokeScaleEnabled={false}
            listening={false}
          />
        </Group>
      )}
    </Group>
  );
};

export default memo(IAICanvasStagingArea);
