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
      shouldShowBrush,
    } = inpainting;

    return {
      cursorPosition,
      width,
      height,
      shouldShowBrushPreview,
      brushSize,
      strokeWidth: 1 / stageScale, // scale stroke thickness
      shouldShowBrush,
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
    shouldShowBrush,
  } = useAppSelector(inpaintingCanvasBrushPreviewSelector);

  if (!shouldShowBrush || !(cursorPosition || shouldShowBrushPreview))
    return null;

  return (
    <>
      <Circle
        x={cursorPosition ? cursorPosition.x : width / 2}
        y={cursorPosition ? cursorPosition.y : height / 2}
        radius={brushSize / 2}
        stroke={'rgba(0,0,0,1)'}
        strokeWidth={strokeWidth}
        strokeEnabled={true}
        listening={false}
      />
      <Circle
        x={cursorPosition ? cursorPosition.x : width / 2}
        y={cursorPosition ? cursorPosition.y : height / 2}
        radius={1}
        fill={'rgba(0,0,0,1)'}
        listening={false}
      />
    </>
  );
};
export default InpaintingCanvasBrushPreviewOutline;
