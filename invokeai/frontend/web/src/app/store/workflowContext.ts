import { createContext, useContext } from 'react';

/**
 * Context to track whether we're in a canvas workflow or nodes workflow.
 * This is used by the useAppDispatch hook to inject the appropriate action routing metadata.
 */
export const WorkflowContext = createContext<{ isCanvasWorkflow: boolean } | null>(null);

/**
 * Hook to check if we're in a canvas workflow context.
 */
export const useIsCanvasWorkflow = (): boolean => {
  const context = useContext(WorkflowContext);
  return context?.isCanvasWorkflow ?? false;
};
