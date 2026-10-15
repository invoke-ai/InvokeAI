import type { XYPosition } from '@features/workflow/contracts';
import type { ReactFlowInstance } from '@xyflow/react';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

/** Expose the mounted flow instance to outside-provider actions that insert nodes at viewport center. */

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

export interface WorkflowFitViewTarget {
  id: string;
  position: XYPosition;
}

/** The project workflow a fit request is for; other editors (including the one being replaced) ignore it. */
export interface WorkflowFitViewScope {
  projectId: string;
  workflowId: string;
}

export interface WorkflowFitViewRequestSnapshot {
  /** A pending request to fit the graph once exactly these nodes are mounted and measured (e.g. after a load). */
  request: { nodes: readonly WorkflowFitViewTarget[]; scope: WorkflowFitViewScope; token: number } | null;
}

/** Lives here rather than in the selection store so the library load paths stay out of the editor's startup graph. */
export const workflowFitViewRequestStore = createExternalStore<WorkflowFitViewRequestSnapshot>({ request: null });

registerAccountOwnedResource({
  clear: () => workflowFitViewRequestStore.setSnapshot({ request: null }),
  name: 'workflow-fit-view-request',
});

/** Fits a document whose editor stays mounted across the change; a newly mounted editor fits on its own. */
export const requestWorkflowFitView = (scope: WorkflowFitViewScope, nodes: readonly WorkflowFitViewTarget[]): void => {
  const previousToken = workflowFitViewRequestStore.getSnapshot().request?.token ?? 0;

  workflowFitViewRequestStore.setSnapshot({
    request: {
      nodes: nodes.map(({ id, position }) => ({ id, position })),
      scope: { projectId: scope.projectId, workflowId: scope.workflowId },
      token: previousToken + 1,
    },
  });
};

export const clearWorkflowFitViewRequest = (): void => {
  workflowFitViewRequestStore.setSnapshot({ request: null });
};
