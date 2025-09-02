import { useStore } from '@nanostores/react';
import type { UnknownAction } from '@reduxjs/toolkit';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { $canvasManagers } from 'features/controlLayers/store/ephemeral';
import { selectCanvasInstance } from 'features/controlLayers/store/selectors';
import type { CanvasState } from 'features/controlLayers/store/types';
import { createContext, memo, useCallback, useContext, useMemo } from 'react';
import { assert } from 'tsafe';

// Define the action type that includes canvasId
type CanvasAction = UnknownAction & {
  payload?: { canvasId?: string } & Record<string, unknown>;
};

interface CanvasInstanceContextValue {
  canvasId: string;
  canvasName?: string;
  manager: CanvasManager;
  dispatch: (action: CanvasAction) => void;
  useSelector: <T>(selector: (state: CanvasState) => T) => T;
}

const CanvasInstanceContext = createContext<CanvasInstanceContextValue | null>(null);

export const CanvasInstanceProvider = memo<{
  canvasId: string;
  canvasName?: string;
  children: React.ReactNode;
}>(({ canvasId, canvasName, children }) => {
  const store = useAppStore();
  const canvasManagers = useStore($canvasManagers);
  const manager = canvasManagers.get(canvasId);

  // Enhanced dispatch function that automatically injects canvasId
  const dispatch = useCallback((action: CanvasAction) => {
    // Clone action and inject canvasId into payload
    const actionWithCanvasId = {
      ...action,
      payload: {
        ...(action.payload || {}),
        canvasId,
      },
    };
    store.dispatch(actionWithCanvasId);
  }, [store, canvasId]);

  // Canvas instance-specific selector hook
  const useSelector = useCallback(<T,>(selector: (state: CanvasState) => T): T => {
    return useAppSelector((state) => {
      const canvasInstance = selectCanvasInstance(state, canvasId);
      if (!canvasInstance) {
        // Return a default/empty value - this can be refined based on usage patterns
        // For now, we'll throw as the components should handle this case
        throw new Error(`Canvas instance ${canvasId} not found`);
      }
      return selector(canvasInstance);
    });
  }, [canvasId]);

  // Memoize the context value to prevent unnecessary re-renders
  const value = useMemo(() => {
    if (!manager) {
      return null;
    }
    
    return {
      canvasId,
      canvasName,
      manager,
      dispatch,
      useSelector,
    };
  }, [canvasId, canvasName, manager, dispatch, useSelector]);

  // Don't render children if manager is not available
  if (!value) {
    return null;
  }

  return (
    <CanvasInstanceContext.Provider value={value}>
      {children}
    </CanvasInstanceContext.Provider>
  );
});

CanvasInstanceProvider.displayName = 'CanvasInstanceProvider';

/**
 * Hook to access the canvas instance context. Must be used within a CanvasInstanceProvider.
 * Throws an error if used outside of the provider.
 */
export const useCanvasContext = (): CanvasInstanceContextValue => {
  const context = useContext(CanvasInstanceContext);
  assert(context, 'useCanvasContext must be used within a CanvasInstanceProvider');
  return context;
};

/**
 * Safe hook to access the canvas instance context. Returns null if used outside of the provider.
 * This is useful for components that may be used both inside and outside the provider.
 */
export const useCanvasContextSafe = (): CanvasInstanceContextValue | null => {
  return useContext(CanvasInstanceContext);
};

/**
 * Hook to get a canvas manager for a specific canvas ID.
 * This is useful for accessing managers from outside the context provider.
 */
export const useCanvasManager = (canvasId: string | null | undefined): CanvasManager | null => {
  const canvasManagers = useStore($canvasManagers);
  return canvasId ? canvasManagers.get(canvasId) || null : null;
};