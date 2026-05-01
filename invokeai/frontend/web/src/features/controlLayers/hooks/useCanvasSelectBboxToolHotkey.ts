import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useEffect } from 'react';

export const useCanvasSelectBboxToolHotkey = () => {
  useAssertSingleton(useCanvasSelectBboxToolHotkey.name);
  const canvasManager = useCanvasManager();

  const onKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }
      canvasManager.tool.onBboxToolHotkeyDown(event);
    },
    [canvasManager]
  );

  const onKeyUp = useCallback(
    (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }
      canvasManager.tool.onBboxToolHotkeyUp(event);
    },
    [canvasManager]
  );

  useEffect(() => {
    return () => {
      canvasManager.tool.clearBboxToolHotkey();
    };
  }, [canvasManager]);

  useRegisteredHotkeys({
    id: 'selectBboxTool',
    category: 'canvas',
    callback: onKeyDown,
    options: { keydown: true, keyup: false, preventDefault: true },
    dependencies: [onKeyDown],
  });

  useRegisteredHotkeys({
    id: 'selectBboxTool',
    category: 'canvas',
    callback: onKeyUp,
    options: { keydown: false, keyup: true, preventDefault: true },
    dependencies: [onKeyUp],
  });
};
