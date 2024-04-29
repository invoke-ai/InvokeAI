import { $ctrl, $meta } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $isMoveStageKeyHeld } from 'features/canvas/store/canvasNanostore';
import { setBrushSize, setStageCoordinates, setStageScale } from 'features/canvas/store/canvasSlice';
import { CANVAS_SCALE_BY, MAX_CANVAS_SCALE, MIN_CANVAS_SCALE } from 'features/canvas/util/constants';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { clamp } from 'lodash-es';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

export const calculateNewBrushSize = (brushSize: number, delta: number) => {
  // This equation was derived by fitting a curve to the desired brush sizes and deltas
  // see https://github.com/invoke-ai/InvokeAI/pull/5542#issuecomment-1915847565
  const targetDelta = Math.sign(delta) * 0.7363 * Math.pow(1.0394, brushSize);
  // This needs to be clamped to prevent the delta from getting too large
  const finalDelta = clamp(targetDelta, -20, 20);
  // The new brush size is also clamped to prevent it from getting too large or small
  const newBrushSize = clamp(brushSize + finalDelta, 1, 500);

  return newBrushSize;
};

const useCanvasWheel = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const stageScale = useAppSelector((s) => s.canvas.stageScale);
  const isMoveStageKeyHeld = useStore($isMoveStageKeyHeld);
  const brushSize = useAppSelector((s) => s.canvas.brushSize);
  const shouldInvertBrushSizeScrollDirection = useAppSelector((s) => s.canvas.shouldInvertBrushSizeScrollDirection);

  return useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      // stop default scrolling
      if (!stageRef.current || isMoveStageKeyHeld) {
        return;
      }

      e.evt.preventDefault();

      // checking for ctrl key is pressed or not,
      // so that brush size can be controlled using ctrl + scroll up/down

      // Invert the delta if the property is set to true
      let delta = e.evt.deltaY;
      if (shouldInvertBrushSizeScrollDirection) {
        delta = -delta;
      }

      if ($ctrl.get() || $meta.get()) {
        dispatch(setBrushSize(calculateNewBrushSize(brushSize, delta)));
      } else {
        const cursorPos = stageRef.current.getPointerPosition();
        let delta = e.evt.deltaY;

        if (!cursorPos) {
          return;
        }

        const mousePointTo = {
          x: (cursorPos.x - stageRef.current.x()) / stageScale,
          y: (cursorPos.y - stageRef.current.y()) / stageScale,
        };
        // when we zoom on trackpad, e.evt.ctrlKey is true
        // in that case lets revert direction
        if (e.evt.ctrlKey) {
          delta = -delta;
        }

        const newScale = clamp(stageScale * CANVAS_SCALE_BY ** delta, MIN_CANVAS_SCALE, MAX_CANVAS_SCALE);

        const newCoordinates = {
          x: cursorPos.x - mousePointTo.x * newScale,
          y: cursorPos.y - mousePointTo.y * newScale,
        };

        dispatch(setStageScale(newScale));
        dispatch(setStageCoordinates(newCoordinates));
      }
    },
    [stageRef, isMoveStageKeyHeld, brushSize, dispatch, stageScale, shouldInvertBrushSizeScrollDirection]
  );
};

export default useCanvasWheel;
