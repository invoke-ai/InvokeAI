import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  setIsMovingStage,
  setStageCoordinates,
} from 'features/canvas/store/canvasSlice';
import { KonvaEventObject } from 'konva/lib/Node';
import { isEqual } from 'lodash-es';

import { useCallback } from 'react';

const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { tool, isMovingBoundingBox } = canvas;
    return {
      tool,
      isStaging,
      isMovingBoundingBox,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const useCanvasDrag = () => {
  const dispatch = useAppDispatch();
  const { tool, isStaging, isMovingBoundingBox } = useAppSelector(selector);

  return {
    handleDragStart: useCallback(() => {
      if (!((tool === 'move' || isStaging) && !isMovingBoundingBox)) {
        return;
      }
      dispatch(setIsMovingStage(true));
    }, [dispatch, isMovingBoundingBox, isStaging, tool]),

    handleDragMove: useCallback(
      (e: KonvaEventObject<MouseEvent>) => {
        if (!((tool === 'move' || isStaging) && !isMovingBoundingBox)) {
          return;
        }

        const newCoordinates = { x: e.target.x(), y: e.target.y() };

        dispatch(setStageCoordinates(newCoordinates));
      },
      [dispatch, isMovingBoundingBox, isStaging, tool]
    ),

    handleDragEnd: useCallback(() => {
      if (!((tool === 'move' || isStaging) && !isMovingBoundingBox)) {
        return;
      }
      dispatch(setIsMovingStage(false));
    }, [dispatch, isMovingBoundingBox, isStaging, tool]),
  };
};

export default useCanvasDrag;
