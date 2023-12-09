import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  // addPointToCurrentEraserLine,
  addPointToCurrentLine,
  setIsDrawing,
  setIsMovingStage,
} from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import Konva from 'konva';
import { MutableRefObject, useCallback } from 'react';

const selector = createMemoizedSelector(
  [activeTabNameSelector, stateSelector, isStagingSelector],
  (activeTabName, { canvas }, isStaging) => {
    const { tool, isDrawing } = canvas;
    return {
      tool,
      isDrawing,
      activeTabName,
      isStaging,
    };
  }
);

const useCanvasMouseUp = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>
) => {
  const dispatch = useAppDispatch();
  const { tool, isDrawing, isStaging } = useAppSelector(selector);

  return useCallback(() => {
    if (tool === 'move' || isStaging) {
      dispatch(setIsMovingStage(false));
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
    dispatch(setIsDrawing(false));
  }, [didMouseMoveRef, dispatch, isDrawing, isStaging, stageRef, tool]);
};

export default useCanvasMouseUp;
