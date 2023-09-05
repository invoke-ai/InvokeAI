import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  addPointToCurrentLine,
  setCursorPosition,
} from 'features/canvas/store/canvasSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import Konva from 'konva';
import { Vector2d } from 'konva/lib/types';
import { isEqual } from 'lodash-es';

import { KonvaEventObject } from 'konva/lib/Node';
import { MutableRefObject, useCallback } from 'react';
import getScaledCursorPosition from '../util/getScaledCursorPosition';
import useColorPicker from './useColorUnderCursor';

const selector = createSelector(
  [activeTabNameSelector, canvasSelector, isStagingSelector],
  (activeTabName, canvas, isStaging) => {
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

const useCanvasTouchMove = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>,
  lastCursorPositionRef: MutableRefObject<Vector2d>
) => {
  const dispatch = useAppDispatch();
  const { isDrawing, tool, isStaging } = useAppSelector(selector);
  const { updateColorUnderCursor } = useColorPicker();

  return useCallback(
    (e: KonvaEventObject<TouchEvent>) => {
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
        addPointToCurrentLine({
          points: [scaledCursorPosition.x, scaledCursorPosition.y],
          strokeWidth: e.evt.targetTouches[0]?.force
            ? e.evt.targetTouches[0]?.force
            : 1,
        })
      );
    },
    [
      didMouseMoveRef,
      dispatch,
      isDrawing,
      isStaging,
      lastCursorPositionRef,
      stageRef,
      tool,
      updateColorUnderCursor,
    ]
  );
};

export default useCanvasTouchMove;
