import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/hooks/useCanvasManager';
import type { Tool } from 'features/controlLayers/store/types';
import { computed } from 'nanostores';
import { useCallback } from 'react';

export const useToolIsSelected = (tool: Tool) => {
  const canvasManager = useCanvasManager();
  const isSelected = useStore(computed(canvasManager.tool.$tool, (t) => t === tool));
  return isSelected;
};

export const useSelectTool = (tool: Tool) => {
  const canvasManager = useCanvasManager();
  const setTool = useCallback(() => {
    canvasManager.tool.$tool.set(tool);
  }, [canvasManager.tool.$tool, tool]);
  return setTool;
};
