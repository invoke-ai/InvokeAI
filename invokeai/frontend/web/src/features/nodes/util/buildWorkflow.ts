import { logger } from 'app/logging/logger';
import { NodesState } from '../store/types';
import { Workflow, zWorkflowEdge, zWorkflowNode } from '../types/types';
import { fromZodError } from 'zod-validation-error';
import { parseify } from 'common/util/serialize';
import i18n from 'i18next';

export const buildWorkflow = (nodesState: NodesState): Workflow => {
  const { nodes, edges } = nodesState.present;
  const { workflow: workflowMeta } = nodesState;
  const workflow: Workflow = {
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
