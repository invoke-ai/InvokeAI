import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';

import { selectCanvasSlice } from './canvasSlice';
import type { CanvasImage } from './canvasTypes';
import { isCanvasBaseImage } from './canvasTypes';

export const isStagingSelector = createMemoizedSelector(
  selectCanvasSlice,
  (canvas) =>
    canvas.batchIds.length > 0 ||
    canvas.layerState.stagingArea.images.length > 0
);

export const initialCanvasImageSelector = (
  state: RootState
): CanvasImage | undefined =>
  state.canvas.layerState.objects.find(isCanvasBaseImage);
