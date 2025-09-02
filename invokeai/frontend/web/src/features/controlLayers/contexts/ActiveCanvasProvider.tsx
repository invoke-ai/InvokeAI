import { useAppSelector } from 'app/store/storeHooks';
import { CanvasInstanceProvider } from 'features/controlLayers/contexts/CanvasInstanceContext';
import { selectActiveCanvasId } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

/**
 * Provides the active canvas context to components outside the canvas workspace.
 * This is useful for panels like layers, parameters, etc. that need to interact
 * with the currently active canvas.
 */
export const ActiveCanvasProvider = memo<{ children: React.ReactNode }>(({ children }) => {
  const activeCanvasId = useAppSelector(selectActiveCanvasId);
  
  if (!activeCanvasId) {
    // No active canvas, render children without context
    // Components should use useCanvasContextSafe() to handle this
    return <>{children}</>;
  }
  
  return (
    <CanvasInstanceProvider canvasId={activeCanvasId}>
      {children}
    </CanvasInstanceProvider>
  );
});

ActiveCanvasProvider.displayName = 'ActiveCanvasProvider';