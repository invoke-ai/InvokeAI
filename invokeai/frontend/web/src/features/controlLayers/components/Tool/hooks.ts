import { useStore } from '@nanostores/react';
import { useCanvasManagerSafe } from 'features/controlLayers/hooks/useCanvasManager';
import type { Tool } from 'features/controlLayers/store/types';
import { computed } from 'nanostores';
import { useCallback } from 'react';

export const useToolIsSelected = (tool: Tool) => {
  const canvasManager = useCanvasManagerSafe();
  if (!canvasManager) {
    return false;
  }
  const isSelected = useStore(computed(canvasManager.tool.$tool, (t) => t === tool));
  return isSelected;
};

export const useSelectTool = (tool: Tool) => {
  const canvasManager = useCanvasManagerSafe();
  const setTool = useCallback(() => {
    if (canvasManager) {
      canvasManager.tool.$tool.set(tool);
    }
  }, [canvasManager, tool]);
  return setTool;
};
