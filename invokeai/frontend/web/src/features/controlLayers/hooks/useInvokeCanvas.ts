import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { canvasManagerFactory } from 'features/controlLayers/konva/CanvasManagerFactory';
import { $canvasManagers } from 'features/controlLayers/store/ephemeral';
import { selectActiveCanvasId } from 'features/controlLayers/store/selectors';
import Konva from 'konva';
import { useLayoutEffect, useState } from 'react';
import { $socket } from 'services/events/stores';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';

const log = logger('canvas');

// This will log warnings when layers > 5
Konva.showWarnings = import.meta.env.MODE === 'development';

const useKonvaPixelRatioWatcher = () => {
  useAssertSingleton('useKonvaPixelRatioWatcher');

  const dpr = useDevicePixelRatio({ round: false });

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);
};

export const useInvokeCanvas = (): ((el: HTMLDivElement | null) => void) => {
  useAssertSingleton('useInvokeCanvas');
  useKonvaPixelRatioWatcher();
  const store = useAppStore();
  const socket = useStore($socket);
  const [container, containerRef] = useState<HTMLDivElement | null>(null);

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

    // TODO: This is a temporary compatibility layer for Phase 2
    // In Phase 3, this will be replaced with proper multi-instance support
    const activeCanvasId = selectActiveCanvasId(store.getState());
    
    // For now, use a default canvas ID if none is active
    const canvasId = activeCanvasId || 'default-canvas';
    
    // Check if we already have a manager for this canvas
    const canvasManagers = $canvasManagers.get();
    const existingManager = canvasManagers.get(canvasId);
    
    if (existingManager) {
      existingManager.stage.setContainer(container);
      return () => {};
    }

    // Create new manager using the factory
    const manager = canvasManagerFactory.createInstance(canvasId, container, store, socket);
    manager.initialize();
    
    // Update the registry
    const updatedManagers = new Map(canvasManagers);
    updatedManagers.set(canvasId, manager);
    $canvasManagers.set(updatedManagers);

    return () => {
      canvasManagerFactory.destroyInstance(canvasId);
      const managers = $canvasManagers.get();
      const newManagers = new Map(managers);
      newManagers.delete(canvasId);
      $canvasManagers.set(newManagers);
    };
  }, [container, socket, store]);

  return containerRef;
};
