import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  $isDrawing,
  setCursorPosition,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  addPointToCurrentLine,
  selectCanvasSlice,
} from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

import useColorPicker from './useColorUnderCursor';

const selector = createMemoizedSelector(
  [activeTabNameSelector, selectCanvasSlice, isStagingSelector],
  (activeTabName, canvas, isStaging) => {
    return {
      tool: canvas.tool,
      activeTabName,
      isStaging,
    };
  }
);

const useCanvasMouseMove = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>,
  lastCursorPositionRef: MutableRefObject<Vector2d>
) => {
  const dispatch = useAppDispatch();
  const isDrawing = useStore($isDrawing);
  const { tool, isStaging } = useAppSelector(selector);
  const { updateColorUnderCursor } = useColorPicker();

  return useCallback(() => {
    if (!stageRef.current) {
      return;
    }

    const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

    if (!scaledCursorPosition) {
      return;
    }

    setCursorPosition(scaledCursorPosition);

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
