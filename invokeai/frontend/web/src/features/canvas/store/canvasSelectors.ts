import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { RootState, stateSelector } from 'app/store/store';
import { CanvasImage, isCanvasBaseImage } from './canvasTypes';

export const isStagingSelector = createMemoizedSelector(
  [stateSelector],
  ({ canvas }) =>
    canvas.batchIds.length > 0 ||
    canvas.layerState.stagingArea.images.length > 0
);

export const initialCanvasImageSelector = (
  state: RootState
): CanvasImage | undefined =>
  state.canvas.layerState.objects.find(isCanvasBaseImage);
