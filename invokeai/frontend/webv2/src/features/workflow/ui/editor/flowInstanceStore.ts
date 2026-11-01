import type { XYPosition } from '@features/workflow/contracts';
import type { ReactFlowInstance } from '@xyflow/react';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

/** Expose the mounted flow instance to outside-provider actions that insert nodes at the editor's center. */

export type WorkflowFlowInstance = ReactFlowInstance<WorkflowFlowNode, WorkflowFlowEdge>;

/**
 * How far the editor toolbar reaches into the graph from the editor's left edge (its 8px inset, its 38px column, and
 * a gap). Graph content a view is fitted to, or a fresh view opens on, starts right of it, so the toolbar never
 * covers a node.
 */
export const WORKFLOW_TOOLBAR_CLEARANCE_PX = 56;

/** Every fit of the editor's graph: XYFlow's 10% padding, with the toolbar's column kept clear on the left. */
export const WORKFLOW_FIT_PADDING = {
  bottom: 0.1,
  left: `${WORKFLOW_TOOLBAR_CLEARANCE_PX}px`,
  right: 0.1,
  top: 0.1,
} as const;

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

let viewElement: HTMLElement | null = null;

/** A callback ref for the editor's on-screen frame, so actions outside the flow can place nodes in what it shows. */
export const registerWorkflowViewElement = (element: HTMLElement | null): (() => void) | undefined => {
  if (!element) {
    return undefined;
  }

  viewElement = element;

  return () => {
    if (viewElement === element) {
      viewElement = null;
    }
  };
};

/** The graph position at the middle of the editor as the user sees it, or null while no editor is shown. */
export const getWorkflowViewCenter = (): XYPosition | null => {
  if (!flowInstance || !viewElement?.isConnected) {
    return null;
  }

  const rect = viewElement.getBoundingClientRect();

  return flowInstance.screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
};

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
