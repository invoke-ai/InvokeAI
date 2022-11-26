import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import { canvasSelector, isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  addLine,
  setIsDrawing,
} from 'features/canvas/store/canvasSlice';
import getScaledCursorPosition from '../util/getScaledCursorPosition';

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

const useCanvasMouseEnter = (
  stageRef: MutableRefObject<Konva.Stage | null>
) => {
  const dispatch = useAppDispatch();
  const { tool, isStaging } = useAppSelector(selector);

  return useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (e.evt.buttons !== 1) return;

      if (!stageRef.current) return;

      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (!scaledCursorPosition || tool === 'move' || isStaging) return;

      dispatch(setIsDrawing(true));

      // Add a new line starting from the current cursor position.
      dispatch(addLine([scaledCursorPosition.x, scaledCursorPosition.y]));
    },
    [stageRef, tool, isStaging, dispatch]
  );
};

export default useCanvasMouseEnter;
