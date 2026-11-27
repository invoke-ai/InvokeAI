import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';

export const useCanvasToggleBboxHotkey = () => {
  const canvasManager = useCanvasManager();

  const handleToggleBboxVisibility = useCallback(() => {
    canvasManager.tool.tools.bbox.toggleBboxVisibility();
  }, [canvasManager]);

  useRegisteredHotkeys({
    id: 'toggleBbox',
    category: 'canvas',
    callback: handleToggleBboxVisibility,
    dependencies: [handleToggleBboxVisibility],
  });
};
