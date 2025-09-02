import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { listenerMiddleware } from 'app/store/store';
import { selectCanvasInstance } from 'features/controlLayers/store/selectors';
import type { Logger } from 'roarr';
import type { AppSocket } from 'services/events/types';

import { CanvasManager } from './CanvasManager';

/**
 * Factory class for managing Canvas Manager instances with proper lifecycle management.
 * This factory replaces the singleton pattern with a registry-based approach that supports
 * multiple canvas instances, each with their own state listeners and cleanup procedures.
 */
export class CanvasManagerFactory {
  private readonly log: Logger;
  private managers = new Map<string, CanvasManager>();
  private unsubscribers = new Map<string, () => void>();

  constructor() {
    this.log = logger('canvasManagerFactory');
  }

  /**
   * Creates a new canvas manager instance with proper state listening setup.
   * 
   * @param canvasId - Unique identifier for this canvas instance
   * @param container - HTML div element that will contain the Konva stage
   * @param store - Redux store instance
   * @param socket - WebSocket client for server communication
   * @returns The created CanvasManager instance
   */
  createInstance(
    canvasId: string,
    container: HTMLDivElement,
    store: AppStore,
    socket: AppSocket
  ): CanvasManager {
    this.log.debug({ canvasId }, 'Creating canvas manager instance');

    if (this.managers.has(canvasId)) {
      this.log.warn({ canvasId }, 'Canvas manager already exists, destroying existing instance');
      this.destroyInstance(canvasId);
    }

    // Create the manager instance with the canvasId
    const manager = new CanvasManager(container, store, socket, canvasId);
    this.managers.set(canvasId, manager);

    // Set up state listener for this specific canvas instance
    const listener = listenerMiddleware.startListening({
      predicate: (action, currentState, previousState) => {
        const oldState = selectCanvasInstance(previousState, canvasId);
        const newState = selectCanvasInstance(currentState, canvasId);
        return oldState !== newState;
      },
      effect: (action, listenerApi) => {
        const latestState = selectCanvasInstance(listenerApi.getState(), canvasId);
        if (latestState) {
          manager.onStateUpdated(latestState);
        }
      },
    });

    this.unsubscribers.set(canvasId, listener.unsubscribe);

    this.log.debug({ canvasId, managersCount: this.managers.size }, 'Canvas manager instance created');
    return manager;
  }

  /**
   * Retrieves an existing canvas manager instance by its ID.
   * 
   * @param canvasId - The ID of the canvas manager to retrieve
   * @returns The CanvasManager instance or undefined if not found
   */
  getInstance(canvasId: string): CanvasManager | undefined {
    return this.managers.get(canvasId);
  }

  /**
   * Destroys a canvas manager instance and cleans up all associated resources.
   * This includes unsubscribing from state listeners and calling the manager's destroy method.
   * 
   * @param canvasId - The ID of the canvas manager to destroy
   */
  destroyInstance(canvasId: string): void {
    this.log.debug({ canvasId }, 'Destroying canvas manager instance');

    // Unsubscribe from state listener
    const unsubscriber = this.unsubscribers.get(canvasId);
    if (unsubscriber) {
      unsubscriber();
      this.unsubscribers.delete(canvasId);
    }

    // Destroy the manager and remove from registry
    const manager = this.managers.get(canvasId);
    if (manager) {
      manager.destroy();
      this.managers.delete(canvasId);
    }

    this.log.debug({ canvasId, managersCount: this.managers.size }, 'Canvas manager instance destroyed');
  }

  /**
   * Destroys all canvas manager instances and cleans up all resources.
   * Useful for application shutdown or complete reset scenarios.
   */
  destroyAll(): void {
    this.log.debug({ managersCount: this.managers.size }, 'Destroying all canvas manager instances');
    
    const canvasIds = Array.from(this.managers.keys());
    for (const canvasId of canvasIds) {
      this.destroyInstance(canvasId);
    }
  }

  /**
   * Returns all active canvas IDs.
   * 
   * @returns Array of canvas IDs that have active managers
   */
  getActiveCanvasIds(): string[] {
    return Array.from(this.managers.keys());
  }

  /**
   * Returns the count of active canvas manager instances.
   * 
   * @returns Number of active managers
   */
  getInstanceCount(): number {
    return this.managers.size;
  }
}

// Export a singleton factory instance for global use
export const canvasManagerFactory = new CanvasManagerFactory();