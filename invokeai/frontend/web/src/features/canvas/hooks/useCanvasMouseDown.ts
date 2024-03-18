import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $isDrawing, $isMovingStage, $tool } from 'features/canvas/store/canvasNanostore';
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
  const isStaging = useAppSelector(isStagingSelector);
  const { commitColorUnderCursor } = useColorPicker();

  return useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) {
        return;
      }

      stageRef.current.container().focus();
      const tool = $tool.get();

      if (tool === 'move' || isStaging) {
        $isMovingStage.set(true);
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

      $isDrawing.set(true);

      // Add a new line starting from the current cursor position.
      dispatch(
        addLine({
          points: [scaledCursorPosition.x, scaledCursorPosition.y],
          tool,
        })
      );
    },
    [stageRef, isStaging, dispatch, commitColorUnderCursor]
  );
};

export default useCanvasMouseDown;
