import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { pick } from 'lodash-es';
import { NodesState } from '../store/types';
import { Workflow, isInvocationNode, isNotesNode } from '../types/types';

export const buildWorkflow = (nodesState: NodesState): Workflow => {
  const { workflow: workflowMeta, nodes, edges } = nodesState;
  const workflow: Workflow = {
    ...workflowMeta,
    nodes: [],
    edges: [],
  };

  nodes.forEach((node) => {
    if (!isInvocationNode(node) && !isNotesNode(node)) {
      return;
    }
    workflow.nodes.push(
      pick(node, ['id', 'type', 'position', 'width', 'height', 'data'])
    );
  });

  edges.forEach((edge) => {
    workflow.edges.push(
      pick(edge, [
        'source',
        'sourceHandle',
        'target',
        'targetHandle',
        'id',
        'type',
      ])
    );
  });

  return workflow;
};

export const workflowSelector = createSelector(stateSelector, ({ nodes }) =>
  buildWorkflow(nodes)
);
