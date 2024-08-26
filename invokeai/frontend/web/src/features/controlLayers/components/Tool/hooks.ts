import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { Tool } from 'features/controlLayers/store/types';
import { computed } from 'nanostores';
import { useCallback } from 'react';

export const useToolIsSelected = (tool: Tool) => {
  const canvasManager = useCanvasManager();
  const isSelected = useStore(computed(canvasManager.stateApi.$tool, (t) => t === tool));
  return isSelected;
};

export const useSelectTool = (tool: Tool) => {
  const canvasManager = useCanvasManager();
  const setTool = useCallback(() => {
    canvasManager.stateApi.$tool.set(tool);
  }, [canvasManager.stateApi.$tool, tool]);
  return setTool;
};
