import { useLayoutEffect } from 'react';
import { RootState, useAppSelector } from '../../../../app/store';
import { maskLayerRef } from '../InpaintingCanvas';

/**
 * Konva's cache() method basically rasterizes an object/canvas.
 * This is needed to rasterize the mask, before setting the opacity.
 * If we do not cache the maskLayer, the brush strokes will have opacity
 * set individually.
 *
 * This logical component simply uses useLayoutEffect() to synchronously
 * cache the mask layer every time something that changes how it should draw
 * is changed.
 */
const Cacher = () => {
  const {
    tool,
    lines,
    cursorPosition,
    brushSize,
    canvasDimensions: { width, height },
    maskColor,
    shouldInvertMask,
    shouldShowMask,
    shouldShowBrushPreview,
    shouldShowCheckboardTransparency,
    imageToInpaint,
    shouldShowBrush,
    shouldShowBoundingBoxFill,
  } = useAppSelector((state: RootState) => state.inpainting);

  useLayoutEffect(() => {
    if (!maskLayerRef.current) return;
    maskLayerRef.current.cache({
      x: 0,
      y: 0,
      width,
      height,
    });
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
  ]);

  return null;
};

export default Cacher;
