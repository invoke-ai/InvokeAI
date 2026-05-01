import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useEffect, useRef } from 'react';

import { beginBboxToolHotkeyPress, endBboxToolHotkeyPress, type BboxToolHotkeyPressedState } from './bboxToolHotkey';

export const useCanvasSelectBboxToolHotkey = () => {
  useAssertSingleton(useCanvasSelectBboxToolHotkey.name);
  const canvasManager = useCanvasManager();
  const pressedStateRef = useRef<BboxToolHotkeyPressedState | null>(null);

  const onKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }

      const started = beginBboxToolHotkeyPress(canvasManager.tool.$tool.get(), Date.now());
      pressedStateRef.current = started.pressedState;

      if (started.nextTool) {
        canvasManager.tool.$tool.set(started.nextTool);
      }
    },
    [canvasManager]
  );

  const onKeyUp = useCallback(
    (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }

      const ended = endBboxToolHotkeyPress({
        currentTool: canvasManager.tool.$tool.get(),
        pressedState: pressedStateRef.current,
        releasedAt: Date.now(),
      });

      pressedStateRef.current = null;

      if (ended.revertToTool) {
        canvasManager.tool.$tool.set(ended.revertToTool);
      }
    },
    [canvasManager]
  );

  const onWindowBlur = useCallback(() => {
    const pressedState = pressedStateRef.current;

    if (!pressedState) {
      return;
    }

    pressedStateRef.current = null;

    if (canvasManager.tool.$tool.get() === 'bbox') {
      canvasManager.tool.$tool.set(pressedState.previousTool);
    }
  }, [canvasManager]);

  useEffect(() => {
    window.addEventListener('blur', onWindowBlur);

    return () => {
      window.removeEventListener('blur', onWindowBlur);
    };
  }, [onWindowBlur]);

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
