import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { memo } from 'react';
import { Group, Rect } from 'react-konva';

const selector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  const {
    boundingBoxCoordinates,
    boundingBoxDimensions,
    stageDimensions,
    stageScale,
    shouldDarkenOutsideBoundingBox,
    stageCoordinates,
  } = canvas;

  return {
    boundingBoxCoordinates,
    boundingBoxDimensions,
    shouldDarkenOutsideBoundingBox,
    stageCoordinates,
    stageDimensions,
    stageScale,
  };
});
const IAICanvasBoundingBoxOverlay = () => {
  const {
    boundingBoxCoordinates,
    boundingBoxDimensions,
    shouldDarkenOutsideBoundingBox,
    stageCoordinates,
    stageDimensions,
    stageScale,
  } = useAppSelector(selector);

  return (
    <Group listening={false}>
      <Rect
        offsetX={stageCoordinates.x / stageScale}
        offsetY={stageCoordinates.y / stageScale}
        height={stageDimensions.height / stageScale}
        width={stageDimensions.width / stageScale}
        fill="rgba(0,0,0,0.4)"
        listening={false}
        visible={shouldDarkenOutsideBoundingBox}
      />
      <Rect
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        fill="rgb(255,255,255)"
        listening={false}
        visible={shouldDarkenOutsideBoundingBox}
        globalCompositeOperation="destination-out"
      />
    </Group>
  );
};

export default memo(IAICanvasBoundingBoxOverlay);
