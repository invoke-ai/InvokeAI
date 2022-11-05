import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { Circle } from 'react-konva';
import { RootState, useAppSelector } from '../../../../app/store';
import { InpaintingState } from '../inpaintingSlice';
import { rgbaColorToRgbString } from '../util/colorToString';

const inpaintingCanvasBrushPreviewSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      cursorPosition,
      canvasDimensions: { width, height },
      brushSize,
      maskColor,
      tool,
      shouldShowBrush,
      isMovingBoundingBox,
      isTransformingBoundingBox,
    } = inpainting;

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
const InpaintingCanvasBrushPreview = () => {
  const {
    cursorPosition,
    width,
    height,
    brushSize,
    maskColorString,
    tool,
    shouldDrawBrushPreview,
  } = useAppSelector(inpaintingCanvasBrushPreviewSelector);

  if (!shouldDrawBrushPreview) return null;

  return (
    <Circle
      x={cursorPosition ? cursorPosition.x : width / 2}
      y={cursorPosition ? cursorPosition.y : height / 2}
      radius={brushSize / 2}
      fill={maskColorString}
      listening={false}
      globalCompositeOperation={
        tool === 'eraser' ? 'destination-out' : 'source-over'
      }
    />
  );
};

export default InpaintingCanvasBrushPreview;
