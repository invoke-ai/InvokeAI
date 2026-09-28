import { useMountEffect } from '@platform/react/useMountEffect';
import { useReactFlow } from '@xyflow/react';

import { getWorkflowViewport } from './workflowViewportStore';

interface WorkflowViewportRestoreRuntimeProps {
  viewportKey: string;
}

/** Reapplies the saved transform after React Flow reconnects effects when its Activity becomes visible. */
export const WorkflowViewportRestoreRuntime = ({ viewportKey }: WorkflowViewportRestoreRuntimeProps) => {
  const flow = useReactFlow();

  useMountEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      const savedViewport = getWorkflowViewport(viewportKey);

      if (!savedViewport) {
        return;
      }

      const currentViewport = flow.getViewport();

      if (
        currentViewport.x === savedViewport.x &&
        currentViewport.y === savedViewport.y &&
        currentViewport.zoom === savedViewport.zoom
      ) {
        return;
      }

      void flow.setViewport(savedViewport, { duration: 0 });
    });

    return () => window.cancelAnimationFrame(frameId);
  });

  return null;
};
