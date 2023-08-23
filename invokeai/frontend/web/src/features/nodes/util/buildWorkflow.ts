import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { NodesState } from '../store/types';
import { Workflow, zWorkflowEdge, zWorkflowNode } from '../types/types';

export const buildWorkflow = (nodesState: NodesState): Workflow => {
  const { workflow: workflowMeta, nodes, edges } = nodesState;
  const workflow: Workflow = {
    ...workflowMeta,
    nodes: [],
    edges: [],
  };

  nodes.forEach((node) => {
    const result = zWorkflowNode.safeParse(node);
    if (!result.success) {
      return;
    }
    workflow.nodes.push(result.data);
  });

  edges.forEach((edge) => {
    const result = zWorkflowEdge.safeParse(edge);
    if (!result.success) {
      return;
    }
    workflow.edges.push(result.data);
  });

  return workflow;
};

export const workflowSelector = createSelector(stateSelector, ({ nodes }) =>
  buildWorkflow(nodes)
);
