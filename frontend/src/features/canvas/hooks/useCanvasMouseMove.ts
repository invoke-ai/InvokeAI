import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import { Vector2d } from 'konva/lib/types';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  addPointToCurrentLine,
  currentCanvasSelector,
  GenericCanvasState,
  isStagingSelector,
  setCursorPosition,
} from '../canvasSlice';
import getScaledCursorPosition from '../util/getScaledCursorPosition';

const selector = createSelector(
  [activeTabNameSelector, currentCanvasSelector, isStagingSelector],
  (activeTabName, currentCanvas, isStaging) => {
    const { tool, isDrawing } = currentCanvas;
    return {
      tool,
      isDrawing,
      activeTabName,
      isStaging,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasMouseMove = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>,
  lastCursorPositionRef: MutableRefObject<Vector2d>
) => {
  const dispatch = useAppDispatch();
  const { isDrawing, tool, isStaging } = useAppSelector(selector);

  return useCallback(() => {
    if (!stageRef.current) return;

    const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

    if (!scaledCursorPosition) return;

    dispatch(setCursorPosition(scaledCursorPosition));

    lastCursorPositionRef.current = scaledCursorPosition;

    if (!isDrawing || tool === 'move' || isStaging) return;

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
  ]);
};

export default useCanvasMouseMove;
