import { createSelector } from '@reduxjs/toolkit';
import { GroupConfig } from 'konva/lib/Group';
import _ from 'lodash';
import { Circle, Group } from 'react-konva';
import { RootState, useAppSelector } from '../../../../app/store';
import { InpaintingState } from '../inpaintingSlice';
import { rgbaColorToRgbString } from '../util/colorToString';

const inpaintingCanvasBrushPreviewSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      cursorPosition,
      stageDimensions: { width, height },
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
const InpaintingCanvasBrushPreview = (props: GroupConfig) => {
  const { ...rest } = props;
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
    <Group {...rest}>
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
    </Group>
  );
};

export default InpaintingCanvasBrushPreview;
