import { logger } from 'app/logging/logger';
import { NodesState } from 'features/nodes/store/types';
import {
  WorkflowV2,
  zWorkflowEdge,
  zWorkflowNode,
} from 'features/nodes/types/workflow';
import { fromZodError } from 'zod-validation-error';
import { parseify } from 'common/util/serialize';
import i18n from 'i18next';

export const buildWorkflow = (nodesState: NodesState): WorkflowV2 => {
  const { workflow: workflowMeta, nodes, edges } = nodesState;
  const workflow: WorkflowV2 = {
    ...workflowMeta,
    nodes: [],
    edges: [],
  };

  nodes
    .filter((n) =>
      ['invocation', 'notes'].includes(n.type ?? '__UNKNOWN_NODE_TYPE__')
    )
    .forEach((node) => {
      const result = zWorkflowNode.safeParse(node);
      if (!result.success) {
        const { message } = fromZodError(result.error, {
          prefix: i18n.t('nodes.unableToParseNode'),
        });
        logger('nodes').warn({ node: parseify(node) }, message);
        return;
      }
      workflow.nodes.push(result.data);
    });

  edges.forEach((edge) => {
    const result = zWorkflowEdge.safeParse(edge);
    if (!result.success) {
      const { message } = fromZodError(result.error, {
        prefix: i18n.t('nodes.unableToParseEdge'),
      });
      logger('nodes').warn({ edge: parseify(edge) }, message);
      return;
    }
    workflow.edges.push(result.data);
  });

  return workflow;
};
