import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import type { NodesState, WorkflowsState } from 'features/nodes/store/types';
import { isInvocationNode, isNotesNode } from 'features/nodes/types/invocation';
import type { WorkflowV2 } from 'features/nodes/types/workflow';
import { zWorkflowEdge, zWorkflowNode } from 'features/nodes/types/workflow';
import i18n from 'i18n';
import { cloneDeep, omit } from 'lodash-es';
import { fromZodError } from 'zod-validation-error';

export type BuildWorkflowArg = {
  nodes: NodesState['nodes'];
  edges: NodesState['edges'];
  workflow: WorkflowsState;
};

export type BuildWorkflowFunction = (arg: BuildWorkflowArg) => WorkflowV2;

export const buildWorkflow: BuildWorkflowFunction = ({
  nodes,
  edges,
  workflow,
}) => {
  const clonedWorkflow = omit(cloneDeep(workflow), 'isTouched');
  const clonedNodes = cloneDeep(nodes);
  const clonedEdges = cloneDeep(edges);

  const newWorkflow: WorkflowV2 = {
    ...clonedWorkflow,
    nodes: [],
    edges: [],
  };

  clonedNodes
    .filter((n) => isInvocationNode(n) || isNotesNode(n)) // Workflows only contain invocation and notes nodes
    .forEach((node) => {
      const result = zWorkflowNode.safeParse(node);
      if (!result.success) {
        const { message } = fromZodError(result.error, {
          prefix: i18n.t('nodes.unableToParseNode'),
        });
        logger('nodes').warn({ node: parseify(node) }, message);
        return;
      }
      newWorkflow.nodes.push(result.data);
    });

  clonedEdges.forEach((edge) => {
    const result = zWorkflowEdge.safeParse(edge);
    if (!result.success) {
      const { message } = fromZodError(result.error, {
        prefix: i18n.t('nodes.unableToParseEdge'),
      });
      logger('nodes').warn({ edge: parseify(edge) }, message);
      return;
    }
    newWorkflow.edges.push(result.data);
  });

  return newWorkflow;
};
