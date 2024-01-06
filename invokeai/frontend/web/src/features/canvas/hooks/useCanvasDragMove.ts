import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  $isMovingBoundingBox,
  setIsMovingStage,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { setStageCoordinates } from 'features/canvas/store/canvasSlice';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';

const useCanvasDrag = () => {
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(isStagingSelector);
  const tool = useAppSelector((s) => s.canvas.tool);
  const isMovingBoundingBox = useStore($isMovingBoundingBox);
  const handleDragStart = useCallback(() => {
    if (!((tool === 'move' || isStaging) && !isMovingBoundingBox)) {
      return;
    }
    setIsMovingStage(true);
  }, [isMovingBoundingBox, isStaging, tool]);

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
    setIsMovingStage(false);
  }, [isMovingBoundingBox, isStaging, tool]);

  return {
    handleDragStart,
    handleDragMove,
    handleDragEnd,
  };
};

export default useCanvasDrag;
