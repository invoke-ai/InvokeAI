import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  addLine,
  setIsDrawing,
  setIsMovingStage,
} from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

import useColorPicker from './useColorUnderCursor';

const selector = createMemoizedSelector(
  [activeTabNameSelector, stateSelector, isStagingSelector],
  (activeTabName, { canvas }, isStaging) => {
    const { tool } = canvas;
    return {
      tool,
      activeTabName,
      isStaging,
    };
  }
);

const useCanvasMouseDown = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const { tool, isStaging } = useAppSelector(selector);
  const { commitColorUnderCursor } = useColorPicker();

  return useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) {
        return;
      }

      stageRef.current.container().focus();

      if (tool === 'move' || isStaging) {
        dispatch(setIsMovingStage(true));
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

      dispatch(setIsDrawing(true));

      // Add a new line starting from the current cursor position.
      dispatch(addLine([scaledCursorPosition.x, scaledCursorPosition.y]));
    },
    [stageRef, tool, isStaging, dispatch, commitColorUnderCursor]
  );
};

export default useCanvasMouseDown;
