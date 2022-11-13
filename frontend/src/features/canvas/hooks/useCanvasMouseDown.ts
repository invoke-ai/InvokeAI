import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  addLine,
  currentCanvasSelector,
  isStagingSelector,
  setIsDrawing,
  setIsMovingStage,
} from '../canvasSlice';
import getScaledCursorPosition from '../util/getScaledCursorPosition';

const selector = createSelector(
  [activeTabNameSelector, currentCanvasSelector, isStagingSelector],
  (activeTabName, currentCanvas, isStaging) => {
    const { tool } = currentCanvas;
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

  return useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!stageRef.current) return;

      stageRef.current.container().focus();

      if (tool === 'move' || isStaging) {
        dispatch(setIsMovingStage(true));
        return;
      }

      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (!scaledCursorPosition) return;

      e.evt.preventDefault();

      dispatch(setIsDrawing(true));

      // Add a new line starting from the current cursor position.
      dispatch(addLine([scaledCursorPosition.x, scaledCursorPosition.y]));
    },
    [stageRef, tool, isStaging, dispatch]
  );
};

export default useCanvasMouseDown;
