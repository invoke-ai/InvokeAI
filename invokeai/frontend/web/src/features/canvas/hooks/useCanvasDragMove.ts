import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  setIsMovingStage,
  setStageCoordinates,
} from 'features/canvas/store/canvasSlice';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';

const useCanvasDrag = () => {
  const dispatch = useAppDispatch();
  const tool = useAppSelector((state) => state.canvas.tool);
  const isMovingBoundingBox = useAppSelector(
    (state) => state.canvas.isMovingBoundingBox
  );
  const isStaging = useAppSelector(isStagingSelector);

  const handleDragStart = useCallback(() => {
    if (!((tool === 'move' || isStaging) && !isMovingBoundingBox)) {
      return;
    }
    dispatch(setIsMovingStage(true));
  }, [dispatch, isMovingBoundingBox, isStaging, tool]);

  const handleDragMove = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!((tool === 'move' || isStaging) && !isMovingBoundingBox)) {
        return;
      }

      const newCoordinates = { x: e.target.x(), y: e.target.y() };

      dispatch(setStageCoordinates(newCoordinates));
    },
    [dispatch, isMovingBoundingBox, isStaging, tool]
  );

  const handleDragEnd = useCallback(() => {
    if (!((tool === 'move' || isStaging) && !isMovingBoundingBox)) {
      return;
    }
    dispatch(setIsMovingStage(false));
  }, [dispatch, isMovingBoundingBox, isStaging, tool]);

  return { handleDragStart, handleDragMove, handleDragEnd };
};

export default useCanvasDrag;
