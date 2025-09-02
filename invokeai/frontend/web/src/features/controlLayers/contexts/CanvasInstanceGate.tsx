import { useCanvasContextSafe } from 'features/controlLayers/contexts/CanvasInstanceContext';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

/**
 * A gate component that only renders its children if the canvas instance context
 * has a manager available. This prevents components from trying to use the manager
 * before it's initialized.
 */
export const CanvasInstanceGate = memo(({ children }: PropsWithChildren) => {
  const context = useCanvasContextSafe();
  
  // Don't render children if there's no context or no manager
  if (!context || !context.manager) {
    return null;
  }
  
  return <>{children}</>;
});

CanvasInstanceGate.displayName = 'CanvasInstanceGate';