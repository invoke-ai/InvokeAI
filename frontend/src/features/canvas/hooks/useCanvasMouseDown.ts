import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  addLine,
  setIsDrawing,
  setIsMovingStage,
} from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from '../util/getScaledCursorPosition';
import useColorPicker from './useColorUnderCursor';

const selector = createSelector(
  [activeTabNameSelector, canvasSelector, isStagingSelector],
  (activeTabName, canvas, isStaging) => {
    const { tool } = canvas;
    return {
      tool,
      activeTabName,
      isStaging,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasMouseDown = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const { tool, isStaging } = useAppSelector(selector);
  const { commitColorUnderCursor } = useColorPicker();

  return useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) return;

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

      if (!scaledCursorPosition) return;

      e.evt.preventDefault();

      dispatch(setIsDrawing(true));

      // Add a new line starting from the current cursor position.
      dispatch(addLine([scaledCursorPosition.x, scaledCursorPosition.y]));
    },
    [stageRef, tool, isStaging, dispatch, commitColorUnderCursor]
  );
};

export default useCanvasMouseDown;
