import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { memo } from 'react';
import { Group, Rect } from 'react-konva';

const selector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  const { boundingBoxCoordinates, boundingBoxDimensions, stageDimensions, stageCoordinates } = canvas;

  return {
    boundingBoxCoordinates,
    boundingBoxDimensions,
    stageCoordinates,
    stageDimensions,
  };
});

const IAICanvasBoundingBoxOverlay = () => {
  const { boundingBoxCoordinates, boundingBoxDimensions, stageCoordinates, stageDimensions } = useAppSelector(selector);
  const shouldDarkenOutsideBoundingBox = useAppSelector((s) => s.canvas.shouldDarkenOutsideBoundingBox);
  const stageScale = useAppSelector((s) => s.canvas.stageScale);

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
