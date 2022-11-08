import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  addPointToCurrentEraserLine,
  addPointToCurrentLine,
  currentCanvasSelector,
  GenericCanvasState,
  setIsDrawing,
} from '../canvasSlice';
import getScaledCursorPosition from '../util/getScaledCursorPosition';

const selector = createSelector(
  [activeTabNameSelector, currentCanvasSelector],
  (activeTabName, canvas: GenericCanvasState) => {
    const {
      isMoveStageKeyHeld,
      isTransformingBoundingBox,
      isMovingBoundingBox,
      tool,
      isDrawing,
    } = canvas;
    return {
      isMoveStageKeyHeld,
      isModifyingBoundingBox: isTransformingBoundingBox || isMovingBoundingBox,
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
  const { isMoveStageKeyHeld, isModifyingBoundingBox, tool, isDrawing } =
    useAppSelector(selector);

  return useCallback(() => {
    if (!didMouseMoveRef.current && isDrawing && stageRef.current) {
      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (!scaledCursorPosition || isModifyingBoundingBox || isMoveStageKeyHeld)
        return;

      /**
       * Extend the current line.
       * In this case, the mouse didn't move, so we append the same point to
       * the line's existing points. This allows the line to render as a circle
       * centered on that point.
       */
      if (tool === 'imageEraser') {
        dispatch(
          addPointToCurrentEraserLine([
            scaledCursorPosition.x,
            scaledCursorPosition.y,
          ])
        );
      } else {
        dispatch(
          addPointToCurrentLine([
            scaledCursorPosition.x,
            scaledCursorPosition.y,
          ])
        );
      }
    } else {
      didMouseMoveRef.current = false;
    }
    dispatch(setIsDrawing(false));
  }, [
    didMouseMoveRef,
    dispatch,
    isDrawing,
    isModifyingBoundingBox,
    isMoveStageKeyHeld,
    stageRef,
    tool,
  ]);
};

export default useCanvasMouseUp;
