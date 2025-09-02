import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { useCanvasContext } from 'features/controlLayers/contexts/CanvasInstanceContext';
import { canvasManagerFactory } from 'features/controlLayers/konva/CanvasManagerFactory';
import { $canvasManagers } from 'features/controlLayers/store/ephemeral';
import Konva from 'konva';
import { useLayoutEffect, useState } from 'react';
import { $isConnected, $socket } from 'services/events/stores';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';

const log = logger('canvas');

// This will log warnings when layers > 5
Konva.showWarnings = import.meta.env.MODE === 'development';

const useKonvaPixelRatioWatcher = () => {
  // No longer a singleton - we support multiple canvas instances
  const dpr = useDevicePixelRatio({ round: false });

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);
};

export const useInvokeCanvas = (): ((el: HTMLDivElement | null) => void) => {
  // No longer a singleton - we support multiple canvas instances
  useKonvaPixelRatioWatcher();
  const store = useAppStore();
  const socket = useStore($socket);
  const isConnected = useStore($isConnected);
  const [container, containerRef] = useState<HTMLDivElement | null>(null);
  const { canvasId } = useCanvasContext();

  useLayoutEffect(() => {
    log.debug({ canvasId }, 'Initializing renderer for canvas');
    console.log('useInvokeCanvas - canvasId:', canvasId, 'container:', container, 'socket:', !!socket, 'isConnected:', isConnected);
    
    if (!container) {
      // Nothing to clean up
      log.debug('No stage container, skipping initialization');
      return () => {};
    }

    if (!socket || !isConnected) {
      log.debug('Socket not connected, skipping initialization');
      console.log('Socket status - exists:', !!socket, 'isConnected:', isConnected);
      return () => {};
    }
    
    // Check if we already have a manager for this canvas
    const canvasManagers = $canvasManagers.get();
    const existingManager = canvasManagers.get(canvasId);
    
    console.log('Canvas managers:', canvasManagers.size, 'Existing manager:', !!existingManager);
    
    if (existingManager) {
      console.log('Reusing existing manager for canvas:', canvasId);
      existingManager.stage.setContainer(container);
      return () => {};
    }

    // Create new manager using the factory
    console.log('Creating new manager for canvas:', canvasId);
    const manager = canvasManagerFactory.createInstance(canvasId, container, store, socket);
    manager.initialize();
    
    // Update the registry
    const updatedManagers = new Map(canvasManagers);
    updatedManagers.set(canvasId, manager);
    $canvasManagers.set(updatedManagers);
    console.log('Manager created and registered for canvas:', canvasId);

    return () => {
      canvasManagerFactory.destroyInstance(canvasId);
      const managers = $canvasManagers.get();
      const newManagers = new Map(managers);
      newManagers.delete(canvasId);
      $canvasManagers.set(newManagers);
    };
  }, [container, socket, store, canvasId, isConnected]);

  return containerRef;
};
