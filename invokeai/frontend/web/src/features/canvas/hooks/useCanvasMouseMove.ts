import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  addPointToCurrentLine,
  setCursorPosition,
} from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import Konva from 'konva';
import { Vector2d } from 'konva/lib/types';
import { isEqual } from 'lodash-es';
import { MutableRefObject, useCallback } from 'react';
import useColorPicker from './useColorUnderCursor';

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
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const useCanvasMouseMove = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>,
  lastCursorPositionRef: MutableRefObject<Vector2d>
) => {
  const dispatch = useAppDispatch();
  const { isDrawing, tool, isStaging } = useAppSelector(selector);
  const { updateColorUnderCursor } = useColorPicker();

  return useCallback(() => {
    if (!stageRef.current) {
      return;
    }

    const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

    if (!scaledCursorPosition) {
      return;
    }

    dispatch(setCursorPosition(scaledCursorPosition));

    lastCursorPositionRef.current = scaledCursorPosition;

    if (tool === 'colorPicker') {
      updateColorUnderCursor();
      return;
    }

    if (!isDrawing || tool === 'move' || isStaging) {
      return;
    }

    didMouseMoveRef.current = true;
    dispatch(
      addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y])
    );
  }, [
    didMouseMoveRef,
    dispatch,
    isDrawing,
    isStaging,
    lastCursorPositionRef,
    stageRef,
    tool,
    updateColorUnderCursor,
  ]);
};

export default useCanvasMouseMove;
