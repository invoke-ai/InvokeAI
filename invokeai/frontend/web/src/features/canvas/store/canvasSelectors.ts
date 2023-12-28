import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import { stateSelector } from 'app/store/store';

import type { CanvasImage } from './canvasTypes';
import { isCanvasBaseImage } from './canvasTypes';

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
