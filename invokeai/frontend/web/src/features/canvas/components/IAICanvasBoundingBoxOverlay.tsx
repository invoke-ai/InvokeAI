import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { isEqual } from 'lodash-es';

import { Group, Rect } from 'react-konva';
import { canvasSelector } from '../store/canvasSelectors';
import { memo } from 'react';

const selector = createSelector(
  canvasSelector,
  (canvas) => {
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
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);
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
    <Group>
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
