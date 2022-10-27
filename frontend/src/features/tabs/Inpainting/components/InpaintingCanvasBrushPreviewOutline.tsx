import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { Circle } from 'react-konva';
import { RootState, useAppSelector } from '../../../../app/store';
import { InpaintingState } from '../inpaintingSlice';

const inpaintingCanvasBrushPreviewSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      cursorPosition,
      canvasDimensions: { width, height },
      shouldShowBrushPreview,
      brushSize,
      stageScale,
    } = inpainting;

    return {
      cursorPosition,
      width,
      height,
      shouldShowBrushPreview,
      brushSize,
      strokeWidth: 1 / stageScale, // scale stroke thickness
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

/**
 * Draws the canvas brush preview outline.
 */
const InpaintingCanvasBrushPreviewOutline = () => {
  const {
    cursorPosition,
    width,
    height,
    shouldShowBrushPreview,
    brushSize,
    strokeWidth,
  } = useAppSelector(inpaintingCanvasBrushPreviewSelector);

  if (!((cursorPosition || shouldShowBrushPreview) && width && height))
    return null;

  return (
    <Circle
      x={cursorPosition ? cursorPosition.x : width / 2}
      y={cursorPosition ? cursorPosition.y : height / 2}
      radius={brushSize / 2}
      stroke={'rgba(0,0,0,1)'}
      strokeWidth={strokeWidth}
      strokeEnabled={true}
      listening={false}
    />
  );
};
export default InpaintingCanvasBrushPreviewOutline;
