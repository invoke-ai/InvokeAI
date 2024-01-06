import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  $isDrawing,
  setIsDrawing,
  setIsMovingStage,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { addPointToCurrentLine } from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import type Konva from 'konva';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

const useCanvasMouseUp = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>
) => {
  const dispatch = useAppDispatch();
  const isDrawing = useStore($isDrawing);
  const tool = useAppSelector((s) => s.canvas.tool);
  const isStaging = useAppSelector(isStagingSelector);

  return useCallback(() => {
    if (tool === 'move' || isStaging) {
      setIsMovingStage(false);
      return;
    }

    if (!didMouseMoveRef.current && isDrawing && stageRef.current) {
      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (!scaledCursorPosition) {
        return;
      }

      /**
       * Extend the current line.
       * In this case, the mouse didn't move, so we append the same point to
       * the line's existing points. This allows the line to render as a circle
       * centered on that point.
       */
      dispatch(
        addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y])
      );
    } else {
      didMouseMoveRef.current = false;
    }
    setIsDrawing(false);
  }, [didMouseMoveRef, dispatch, isDrawing, isStaging, stageRef, tool]);
};

export default useCanvasMouseUp;
