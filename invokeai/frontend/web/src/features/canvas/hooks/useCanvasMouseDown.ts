import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  setIsDrawing,
  setIsMovingStage,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { addLine } from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

import useColorPicker from './useColorUnderCursor';

const useCanvasMouseDown = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const tool = useAppSelector((s) => s.canvas.tool);
  const isStaging = useAppSelector(isStagingSelector);
  const { commitColorUnderCursor } = useColorPicker();

  return useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) {
        return;
      }

      stageRef.current.container().focus();

      if (tool === 'move' || isStaging) {
        setIsMovingStage(true);
        return;
      }

      if (tool === 'colorPicker') {
        commitColorUnderCursor();
        return;
      }

      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (!scaledCursorPosition) {
        return;
      }

      e.evt.preventDefault();

      setIsDrawing(true);

      // Add a new line starting from the current cursor position.
      dispatch(addLine([scaledCursorPosition.x, scaledCursorPosition.y]));
    },
    [stageRef, tool, isStaging, dispatch, commitColorUnderCursor]
  );
};

export default useCanvasMouseDown;
