import { RootState } from 'app/store';
import { CanvasImage, CanvasState, isCanvasBaseImage } from './canvasTypes';

export const canvasSelector = (state: RootState): CanvasState => state.canvas;

export const isStagingSelector = (state: RootState): boolean =>
  state.canvas.layerState.stagingArea.images.length > 0;

export const initialCanvasImageSelector = (
  state: RootState
): CanvasImage | undefined =>
  state.canvas.layerState.objects.find(isCanvasBaseImage);
