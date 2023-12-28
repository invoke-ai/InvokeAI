import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  setIsMovingStage,
  setStageCoordinates,
} from 'features/canvas/store/canvasSlice';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';

const selector = createMemoizedSelector(
  [stateSelector, isStagingSelector],
  ({ canvas }, isStaging) => {
    const { tool, isMovingBoundingBox } = canvas;
    return {
      tool,
      isStaging,
      isMovingBoundingBox,
    };
  }
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
