import { useLayoutEffect } from 'react';
import { RootState, useAppSelector } from '../../../../app/store';
import { maskLayerRef } from '../InpaintingCanvas';

const useCacher = () => {
  const {
    tool,
    lines,
    cursorPosition,
    brushSize,
    stageDimensions: { width, height },
    maskColor,
    shouldInvertMask,
    shouldShowMask,
    shouldShowBrushPreview,
    shouldShowCheckboardTransparency,
    imageToInpaint,
    shouldShowBrush,
    shouldShowBoundingBoxFill,
    shouldLockBoundingBox,
    stageScale,
    pastLines,
    futureLines,
    doesCanvasNeedScaling,
    isDrawing,
    isTransformingBoundingBox,
    isMovingBoundingBox,
    shouldShowBoundingBox,
    stageCoordinates: { x, y },
  } = useAppSelector((state: RootState) => state.inpainting);

  useLayoutEffect(() => {
    if (!maskLayerRef.current) return;
    maskLayerRef.current.cache();
  }, [
    lines,
    cursorPosition,
    width,
    height,
    tool,
    brushSize,
    maskColor,
    shouldInvertMask,
    shouldShowMask,
    shouldShowBrushPreview,
    shouldShowCheckboardTransparency,
    imageToInpaint,
    shouldShowBrush,
    shouldShowBoundingBoxFill,
    shouldShowBoundingBox,
    shouldLockBoundingBox,
    stageScale,
    pastLines,
    futureLines,
    doesCanvasNeedScaling,
    isDrawing,
    isTransformingBoundingBox,
    isMovingBoundingBox,
    x,
    y,
  ]);
};

export default useCacher;
