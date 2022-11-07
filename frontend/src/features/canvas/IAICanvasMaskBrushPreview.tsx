import { createSelector } from '@reduxjs/toolkit';
import { GroupConfig } from 'konva/lib/Group';
import _ from 'lodash';
import { Circle, Group } from 'react-konva';
import { useAppSelector } from 'app/store';
import {
  currentCanvasSelector,
  GenericCanvasState,
} from 'features/canvas/canvasSlice';
import { rgbaColorToRgbString } from './util/colorToString';

const canvasBrushPreviewSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) => {
    const {
      cursorPosition,
      stageDimensions: { width, height },
      toolSize: brushSize,
      maskColor,
      tool,
      shouldShowBrush,
      isMovingBoundingBox,
      isTransformingBoundingBox,
    } = currentCanvas;

    return {
      cursorPosition,
      width,
      height,
      brushSize,
      maskColorString: rgbaColorToRgbString(maskColor),
      tool,
      shouldShowBrush,
      shouldDrawBrushPreview:
        !(
          isMovingBoundingBox ||
          isTransformingBoundingBox ||
          !cursorPosition
        ) && shouldShowBrush,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

/**
 * Draws a black circle around the canvas brush preview.
 */
const IAICanvasBrushPreview = (props: GroupConfig) => {
  const { ...rest } = props;
  const {
    cursorPosition,
    width,
    height,
    brushSize,
    maskColorString,
    tool,
    shouldDrawBrushPreview,
  } = useAppSelector(canvasBrushPreviewSelector);

  if (!shouldDrawBrushPreview) return null;

  return (
    <Group {...rest}>
      <Circle
        x={cursorPosition ? cursorPosition.x : width / 2}
        y={cursorPosition ? cursorPosition.y : height / 2}
        radius={brushSize / 2}
        fill={maskColorString}
        listening={false}
        globalCompositeOperation={
          tool === 'maskEraser' ? 'destination-out' : 'source-over'
        }
      />
    </Group>
  );
};

export default IAICanvasBrushPreview;
