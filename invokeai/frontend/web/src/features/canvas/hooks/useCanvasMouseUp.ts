import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  $isDrawing,
  setIsDrawing,
  setIsMovingStage,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  // addPointToCurrentEraserLine,
  addPointToCurrentLine,
} from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import type Konva from 'konva';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

const selector = createMemoizedSelector(
  [stateSelector, isStagingSelector],
  ({ canvas }, isStaging) => {
    return {
      tool: canvas.tool,
      isStaging,
    };
  }
);

const useCanvasMouseUp = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>
) => {
  const dispatch = useAppDispatch();
  const isDrawing = useStore($isDrawing);
  const { tool, isStaging } = useAppSelector(selector);

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
