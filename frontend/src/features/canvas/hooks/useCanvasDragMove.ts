import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import { KonvaEventObject } from 'konva/lib/Node';
import _ from 'lodash';
import { useCallback } from 'react';
import { canvasSelector, isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  setIsMovingStage,
  setStageCoordinates,
} from 'features/canvas/store/canvasSlice';

const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { tool } = canvas;
    return {
      tool,
      isStaging,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const useCanvasDrag = () => {
  const dispatch = useAppDispatch();
  const { tool, isStaging } = useAppSelector(selector);

  return {
    handleDragStart: useCallback(() => {
      if (!(tool === 'move' || isStaging)) return;
      dispatch(setIsMovingStage(true));
    }, [dispatch, isStaging, tool]),

    handleDragMove: useCallback(
      (e: KonvaEventObject<MouseEvent>) => {
        if (!(tool === 'move' || isStaging)) return;

        const newCoordinates = { x: e.target.x(), y: e.target.y() };

        dispatch(setStageCoordinates(newCoordinates));
      },
      [dispatch, isStaging, tool]
    ),

    handleDragEnd: useCallback(() => {
      if (!(tool === 'move' || isStaging)) return;
      dispatch(setIsMovingStage(false));
    }, [dispatch, isStaging, tool]),
  };
};

export default useCanvasDrag;
