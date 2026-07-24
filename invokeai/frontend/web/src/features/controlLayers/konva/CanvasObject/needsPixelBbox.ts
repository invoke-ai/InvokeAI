import type { AnyObjectState } from 'features/controlLayers/konva/CanvasObject/types';

export const objectStateNeedsPixelBbox = (state: AnyObjectState): boolean => {
  if (
    state.type === 'eraser_line' ||
    state.type === 'eraser_line_with_pressure' ||
    state.type === 'image' ||
    state.type === 'brush_line' ||
    state.type === 'brush_line_with_pressure' ||
    state.type === 'rect' ||
    state.type === 'oval' ||
    state.type === 'polygon' ||
    state.type === 'lasso' ||
    state.type === 'gradient'
  ) {
    if (state.type === 'image') {
      return state.usePixelBbox !== false;
    }

    if (state.type === 'eraser_line' || state.type === 'eraser_line_with_pressure') {
      return true;
    }

    if (state.type === 'brush_line' || state.type === 'brush_line_with_pressure') {
      return Boolean(state.clip);
    }

    if (state.type === 'rect' || state.type === 'oval') {
      return state.compositeOperation !== 'source-over' || Boolean(state.clip);
    }

    if (state.type === 'polygon' || state.type === 'lasso') {
      return state.compositeOperation !== 'source-over';
    }

    return state.globalCompositeOperation === 'source-atop';
  }

  return false;
};
