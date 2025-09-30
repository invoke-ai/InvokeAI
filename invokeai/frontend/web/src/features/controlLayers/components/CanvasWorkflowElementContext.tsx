import { useAppSelector } from 'app/store/storeHooks';
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

  const value = useMemo<CanvasWorkflowElementContextValue>(
    () => ({
      getElement: (id: string) => nodesState.form.elements[id],
    }),
    [nodesState.form.elements]
  );

  return <CanvasWorkflowElementContext.Provider value={value}>{children}</CanvasWorkflowElementContext.Provider>;
});
CanvasWorkflowElementProvider.displayName = 'CanvasWorkflowElementProvider';

/**
 * Hook to get an element, using canvas workflow context if available,
 * otherwise falls back to regular nodes.
 */
export const useCanvasWorkflowElement = (): ((id: string) => FormElement | undefined) | null => {
  return useContext(CanvasWorkflowElementContext)?.getElement ?? null;
};
