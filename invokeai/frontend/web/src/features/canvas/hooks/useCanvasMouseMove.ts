import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $cursorPosition, $isDrawing, $tool } from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { addPointToCurrentLine } from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

import useColorPicker from './useColorUnderCursor';

const useCanvasMouseMove = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>,
  lastCursorPositionRef: MutableRefObject<Vector2d>
) => {
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(isStagingSelector);
  const { updateColorUnderCursor } = useColorPicker();

  return useCallback(() => {
    if (!stageRef.current) {
      return;
    }

    const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

    if (!scaledCursorPosition) {
      return;
    }

    $cursorPosition.set(scaledCursorPosition);

    lastCursorPositionRef.current = scaledCursorPosition;
    const tool = $tool.get();

    if (tool === 'colorPicker') {
      updateColorUnderCursor();
      return;
    }

    if (!$isDrawing.get() || tool === 'move' || isStaging) {
      return;
    }

    didMouseMoveRef.current = true;
    dispatch(addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y]));
  }, [didMouseMoveRef, dispatch, isStaging, lastCursorPositionRef, stageRef, updateColorUnderCursor]);
};

export default useCanvasMouseMove;
