import { useAppSelector } from 'app/store/storeHooks';
import { WorkflowContext } from 'app/store/workflowContext';
import { selectCanvasWorkflowNodesSlice } from 'features/controlLayers/store/canvasWorkflowNodesSlice';
import type { FormElement } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';

/**
 * Context that provides element lookup from canvas workflow nodes instead of regular nodes.
 * This ensures that when viewing canvas workflow fields, we read from the shadow slice.
 */

type CanvasWorkflowElementContextValue = {
  getElement: (id: string) => FormElement | undefined;
};

const CanvasWorkflowElementContext = createContext<CanvasWorkflowElementContextValue | null>(null);

export const CanvasWorkflowElementProvider = memo(({ children }: PropsWithChildren) => {
  const nodesState = useAppSelector(selectCanvasWorkflowNodesSlice);

  const elementValue = useMemo<CanvasWorkflowElementContextValue>(
    () => ({
      getElement: (id: string) => nodesState.form.elements[id],
    }),
    [nodesState.form.elements]
  );

  const workflowValue = useMemo(() => ({ isCanvasWorkflow: true }), []);

  return (
    <WorkflowContext.Provider value={workflowValue}>
      <CanvasWorkflowElementContext.Provider value={elementValue}>{children}</CanvasWorkflowElementContext.Provider>
    </WorkflowContext.Provider>
  );
});
CanvasWorkflowElementProvider.displayName = 'CanvasWorkflowElementProvider';

/**
 * Hook to get an element, using canvas workflow context if available,
 * otherwise falls back to regular nodes.
 */
export const useCanvasWorkflowElement = (): ((id: string) => FormElement | undefined) | null => {
  return useContext(CanvasWorkflowElementContext)?.getElement ?? null;
};
