import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
import type { WorkflowMode } from 'features/nodes/store/types';
import { createContext, useContext } from 'react';

// Create a context to detect if we're in canvas workflow
const CanvasWorkflowModeContext = createContext<WorkflowMode | null>(null);

export { CanvasWorkflowModeContext };

/**
 * Returns the appropriate workflow mode.
 * If in canvas workflow context, always returns 'view'.
 * Otherwise returns the workflow tab's current mode.
 */
export const useWorkflowMode = (): WorkflowMode => {
  const canvasMode = useContext(CanvasWorkflowModeContext);
  const workflowTabMode = useAppSelector(selectWorkflowMode);

  // If we're in canvas workflow context, use 'view' mode
  if (canvasMode !== null) {
    return canvasMode;
  }

  // Otherwise use the workflow tab's mode
  return workflowTabMode;
};