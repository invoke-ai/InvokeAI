import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import {
  addEraserLine,
  addLine,
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
      toolSize,
    } = canvas;
    return {
      isMoveStageKeyHeld,
      isModifyingBoundingBox: isTransformingBoundingBox || isMovingBoundingBox,
      tool,
      toolSize,
      activeTabName,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasMouseDown = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const dispatch = useAppDispatch();
  const { isMoveStageKeyHeld, isModifyingBoundingBox, tool, toolSize } =
    useAppSelector(selector);

  return useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!stageRef.current) return;

      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (
        !scaledCursorPosition ||
        isModifyingBoundingBox ||
        isMoveStageKeyHeld
      )
        return;
      e.evt.preventDefault();
      dispatch(setIsDrawing(true));
      if (tool === 'imageEraser') {
        // Add a new line starting from the current cursor position.
        dispatch(
          addEraserLine({
            strokeWidth: toolSize / 2,
            points: [scaledCursorPosition.x, scaledCursorPosition.y],
          })
        );
      } else {
        // Add a new line starting from the current cursor position.
        dispatch(
          addLine({
            tool,
            strokeWidth: toolSize / 2,
            points: [scaledCursorPosition.x, scaledCursorPosition.y],
          })
        );
      }
    },
    [
      stageRef,
      isModifyingBoundingBox,
      isMoveStageKeyHeld,
      dispatch,
      tool,
      toolSize,
    ]
  );
};

export default useCanvasMouseDown;
