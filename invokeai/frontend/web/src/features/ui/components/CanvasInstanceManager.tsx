import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { canvasInstanceAdded } from 'features/controlLayers/store/canvasesSlice';
import { selectCanvasCount } from 'features/controlLayers/store/selectors';
import { nanoid } from 'nanoid';
import { memo, useCallback } from 'react';
import { PiPlus } from 'react-icons/pi';

interface CanvasInstanceManagerProps {
  maxCanvases?: number;
}

export const CanvasInstanceManager = memo(({ maxCanvases = 3 }: CanvasInstanceManagerProps) => {
  const dispatch = useAppDispatch();
  const canvasCount = useAppSelector(selectCanvasCount);
  
  const addCanvas = useCallback(() => {
    if (canvasCount >= maxCanvases) {
return;
}
    
    const canvasId = nanoid();
    const canvasName = `Canvas ${canvasCount + 1}`;
    
    // For now, just add to Redux. The dockview panel creation will be handled
    // by other parts of the system that have access to the dockview API
    dispatch(canvasInstanceAdded({ canvasId, name: canvasName }));
    
    // TODO: Trigger panel creation through a global event or state change
    // that the dockview can listen to
  }, [canvasCount, maxCanvases, dispatch]);
  
  const canCanAddCanvas = canvasCount < maxCanvases;
  
  if (canvasCount === 0) {
    return null;
  }
  
  return (
    <Flex gap={2} alignItems="center">
      <Text fontSize="sm" color="base.300">
        Canvases: {canvasCount}/{maxCanvases}
      </Text>
      
      {canCanAddCanvas && (
        <IconButton
          aria-label="Add Canvas"
          icon={<PiPlus />}
          size="sm"
          variant="ghost"
          onClick={addCanvas}
          tooltip="Add new canvas"
        />
      )}
    </Flex>
  );
});

CanvasInstanceManager.displayName = 'CanvasInstanceManager';