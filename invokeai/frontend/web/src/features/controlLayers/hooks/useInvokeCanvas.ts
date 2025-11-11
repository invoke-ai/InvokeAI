import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { $canvasManagers } from 'features/controlLayers/store/ephemeral';
import Konva from 'konva';
import { useLayoutEffect, useState } from 'react';
import { $socket } from 'services/events/stores';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';

const log = logger('canvas');

// This will log warnings when layers > 5
Konva.showWarnings = import.meta.env.MODE === 'development';

const useKonvaPixelRatioWatcher = (canvasId: string) => {
  useAssertSingleton(`useKonvaPixelRatioWatcher-${canvasId}`);

  const dpr = useDevicePixelRatio({ round: false });

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);
};

export const useInvokeCanvas = (canvasId: string): ((el: HTMLDivElement | null) => void) => {
  useAssertSingleton(`useInvokeCanvas-${canvasId}`);
  useKonvaPixelRatioWatcher(canvasId);
  const store = useAppStore();
  const socket = useStore($socket);
  const [container, containerRef] = useState<HTMLDivElement | null>(null);
  const currentManager = $canvasManagers.get()[canvasId];

  useLayoutEffect(() => {
    log.debug('Initializing renderer');
    if (!container) {
      // Nothing to clean up
      log.debug('No stage container, skipping initialization');
      return () => {};
    }

    if (!socket) {
      log.debug('Socket not connected, skipping initialization');
      return () => {};
    }

    if (currentManager) {
      currentManager.stage.setContainer(container);
      return;
    }

    const manager = new CanvasManager(canvasId, container, store, socket);
    manager.initialize();

    return () => {
      manager.destroy();
    };
  }, [canvasId, container, socket, store, currentManager]);

  return containerRef;
};
