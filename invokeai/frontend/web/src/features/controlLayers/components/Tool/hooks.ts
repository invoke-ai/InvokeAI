import { useStore } from '@nanostores/react';
import { $false } from 'app/store/nanostores/util';
import { useCanvasManagerSafe } from 'features/controlLayers/hooks/useCanvasManager';
import type { Tool } from 'features/controlLayers/store/types';
import { computed } from 'nanostores';
import { useCallback } from 'react';

export const useToolIsSelected = (tool: Tool) => {
  const canvasManager = useCanvasManagerSafe();
  const isSelected = useStore(canvasManager ? computed(canvasManager.tool.$tool, (t) => t === tool) : $false);
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
