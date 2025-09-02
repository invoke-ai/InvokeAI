import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/hooks/useCanvasManager';
import { fitRectToGrid } from 'features/controlLayers/konva/util';
import { selectMaskBlur } from 'features/controlLayers/store/paramsSlice';
import { useCallback } from 'react';

export const useAutoFitBBoxToMasks = () => {
  const canvasManager = useCanvasManager();
  const maskBlur = useAppSelector(selectMaskBlur);

  const fitBBoxToMasks = useCallback(() => {
    // Get the rect of all visible inpaint masks
    const visibleRect = canvasManager.compositor.getVisibleRectOfType('inpaint_mask');

    // Can't fit the bbox to nothing
    if (visibleRect.height === 0 || visibleRect.width === 0) {
      return;
    }

    // Account for mask blur expansion and add 8px padding
    const padding = 8;
    const totalPadding = maskBlur + padding;

    const expandedRect = {
      x: visibleRect.x - totalPadding,
      y: visibleRect.y - totalPadding,
      width: visibleRect.width + totalPadding * 2,
      height: visibleRect.height + totalPadding * 2,
    };

    // Apply grid fitting using the bbox grid size
    const gridSize = canvasManager.stateApi.getBboxGridSize();
    const rect = fitRectToGrid(expandedRect, gridSize);

    // Update the generation bbox
    canvasManager.stateApi.setGenerationBbox(rect);
  }, [canvasManager, maskBlur]);

  return fitBBoxToMasks;
};
