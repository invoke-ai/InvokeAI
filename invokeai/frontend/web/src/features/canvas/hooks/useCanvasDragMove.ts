import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $isMovingBoundingBox, $isMovingStage, $tool } from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { setStageCoordinates } from 'features/canvas/store/canvasSlice';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';

const useCanvasDrag = () => {
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(isStagingSelector);
  const handleDragStart = useCallback(() => {
    if (!(($tool.get() === 'move' || isStaging) && !$isMovingBoundingBox.get())) {
      return;
    }
    $isMovingStage.set(true);
  }, [isStaging]);

  const handleDragMove = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      const tool = $tool.get();
      if (!((tool === 'move' || isStaging) && !$isMovingBoundingBox.get())) {
        return;
      }

      const newCoordinates = { x: e.target.x(), y: e.target.y() };

      dispatch(setStageCoordinates(newCoordinates));
    },
    [dispatch, isStaging]
  );

  const handleDragEnd = useCallback(() => {
    if (!(($tool.get() === 'move' || isStaging) && !$isMovingBoundingBox.get())) {
      return;
    }
    $isMovingStage.set(false);
  }, [isStaging]);

  return {
    handleDragStart,
    handleDragMove,
    handleDragEnd,
  };
};

export default useCanvasDrag;
