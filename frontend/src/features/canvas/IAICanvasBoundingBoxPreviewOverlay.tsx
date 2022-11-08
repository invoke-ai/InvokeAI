import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { Group, Rect } from 'react-konva';
import { useAppSelector } from 'app/store';
import {} from 'common/util/roundDownToMultiple';
import {
  currentCanvasSelector,
  GenericCanvasState,
} from 'features/canvas/canvasSlice';
import { rgbaColorToString } from './util/colorToString';
import { GroupConfig } from 'konva/lib/Group';

const boundingBoxPreviewSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) => {
    const {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      boundingBoxPreviewFill,
      stageDimensions,
      stageScale,
      stageCoordinates,
    } = currentCanvas;
    return {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      boundingBoxPreviewFillString: rgbaColorToString(boundingBoxPreviewFill),
      stageCoordinates,
      stageDimensions,
      stageScale,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

type IAICanvasBoundingBoxPreviewOverlayProps = GroupConfig;

/**
 * Shades the area around the mask.
 */
const IAICanvasBoundingBoxPreviewOverlay = (
  props: IAICanvasBoundingBoxPreviewOverlayProps
) => {
  const { ...rest } = props;
  const {
    boundingBoxCoordinates,
    boundingBoxDimensions,
    boundingBoxPreviewFillString,
    stageDimensions,
    stageScale,
    stageCoordinates,
  } = useAppSelector(boundingBoxPreviewSelector);
  return (
    <Group listening={false} {...rest}>
      <Rect
        offsetX={stageCoordinates.x / stageScale}
        offsetY={stageCoordinates.y / stageScale}
        height={stageDimensions.height / stageScale}
        width={stageDimensions.width / stageScale}
        fill={boundingBoxPreviewFillString}
        listening={false}
      />
      <Rect
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        fill={'rgb(255,255,255)'}
        listening={false}
        globalCompositeOperation={'destination-out'}
      />
    </Group>
  );
};

export default IAICanvasBoundingBoxPreviewOverlay;
