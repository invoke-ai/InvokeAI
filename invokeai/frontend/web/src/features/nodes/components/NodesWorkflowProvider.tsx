import { WorkflowContext } from 'app/store/workflowContext';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';

/**
 * Provider that marks the nodes/workflow editor context.
 * This ensures field actions are routed to the nodes slice, not the canvas workflow slice.
 */
export const NodesWorkflowProvider = memo(({ children }: PropsWithChildren) => {
  const value = useMemo(() => ({ isCanvasWorkflow: false }), []);

  return <WorkflowContext.Provider value={value}>{children}</WorkflowContext.Provider>;
});

NodesWorkflowProvider.displayName = 'NodesWorkflowProvider';
