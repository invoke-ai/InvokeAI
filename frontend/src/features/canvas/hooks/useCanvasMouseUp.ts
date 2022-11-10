import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  // addPointToCurrentEraserLine,
  addPointToCurrentLine,
  currentCanvasSelector,
  GenericCanvasState,
  setIsDrawing,
  setIsMovingStage,
} from '../canvasSlice';
import getScaledCursorPosition from '../util/getScaledCursorPosition';

const selector = createSelector(
  [activeTabNameSelector, currentCanvasSelector],
  (activeTabName, canvas: GenericCanvasState) => {
    const { tool, isDrawing } = canvas;
    return {
      tool,
      isDrawing,
      activeTabName,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasMouseUp = (
  stageRef: MutableRefObject<Konva.Stage | null>,
  didMouseMoveRef: MutableRefObject<boolean>
) => {
  const dispatch = useAppDispatch();
  const { tool, isDrawing } = useAppSelector(selector);

  return useCallback(() => {
    if (tool === 'move') {
      dispatch(setIsMovingStage(false));
      return;
    }

    if (!didMouseMoveRef.current && isDrawing && stageRef.current) {
      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (!scaledCursorPosition) return;

      /**
       * Extend the current line.
       * In this case, the mouse didn't move, so we append the same point to
       * the line's existing points. This allows the line to render as a circle
       * centered on that point.
       */
      dispatch(
        addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y])
      );
    } else {
      didMouseMoveRef.current = false;
    }
    dispatch(setIsDrawing(false));
  }, [didMouseMoveRef, dispatch, isDrawing, stageRef, tool]);
};

export default useCanvasMouseUp;
