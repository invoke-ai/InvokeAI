import type { ReactFlowInstance } from '@xyflow/react';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

/**
 * Module-level handle to the mounted workflow editor's ReactFlow instance, so
 * surfaces outside the ReactFlowProvider (the widget frame's header actions)
 * can place inserted nodes at the current viewport center. Mirrors the legacy
 * editor's nanostore-held instance.
 */

export type WorkflowFlowInstance = ReactFlowInstance<WorkflowFlowNode, WorkflowFlowEdge>;

let flowInstance: WorkflowFlowInstance | null = null;

export const registerWorkflowFlowInstance = (instance: WorkflowFlowInstance): void => {
  flowInstance = instance;
};

export const releaseWorkflowFlowInstance = (instance: WorkflowFlowInstance): void => {
  if (flowInstance === instance) {
    flowInstance = null;
  }
};

export const getWorkflowFlowInstance = (): WorkflowFlowInstance | null => flowInstance;
