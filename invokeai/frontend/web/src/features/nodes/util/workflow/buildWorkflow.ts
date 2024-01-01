import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import type { NodesState, WorkflowsState } from 'features/nodes/store/types';
import { isInvocationNode, isNotesNode } from 'features/nodes/types/invocation';
import { type WorkflowV2, zWorkflowV2 } from 'features/nodes/types/workflow';
import i18n from 'i18n';
import { cloneDeep, omit } from 'lodash-es';
import { fromZodError } from 'zod-validation-error';

export type BuildWorkflowArg = {
  nodes: NodesState['nodes'];
  edges: NodesState['edges'];
  workflow: WorkflowsState;
};

export type BuildWorkflowFunction = (arg: BuildWorkflowArg) => WorkflowV2;

export const buildWorkflowFast: BuildWorkflowFunction = ({
  nodes,
  edges,
  workflow,
}: BuildWorkflowArg): WorkflowV2 => {
  const clonedWorkflow = omit(cloneDeep(workflow), 'isTouched');

  const newWorkflow: WorkflowV2 = {
    ...clonedWorkflow,
    nodes: [],
    edges: [],
  };

  nodes.forEach((node) => {
    if (isInvocationNode(node) && node.type) {
      newWorkflow.nodes.push({
        id: node.id,
        type: node.type,
        data: cloneDeep(node.data),
        position: { ...node.position },
        width: node.width,
        height: node.height,
      });
    } else if (isNotesNode(node) && node.type) {
      newWorkflow.nodes.push({
        id: node.id,
        type: node.type,
        data: cloneDeep(node.data),
        position: { ...node.position },
        width: node.width,
        height: node.height,
      });
    }
  });

  edges.forEach((edge) => {
    if (edge.type === 'default' && edge.sourceHandle && edge.targetHandle) {
      newWorkflow.edges.push({
        id: edge.id,
        type: edge.type,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
      });
    } else if (edge.type === 'collapsed') {
      newWorkflow.edges.push({
        id: edge.id,
        type: edge.type,
        source: edge.source,
        target: edge.target,
      });
    }
  });

  return newWorkflow;
};

export const buildWorkflowRight = ({
  nodes,
  edges,
  workflow,
}: BuildWorkflowArg): WorkflowV2 | null => {
  const newWorkflowUnsafe = {
    ...workflow,
    nodes,
    edges,
  };

  const result = zWorkflowV2.safeParse(newWorkflowUnsafe);

  if (!result.success) {
    const { message } = fromZodError(result.error, {
      prefix: i18n.t('nodes.unableToParseNode'),
    });

    logger('nodes').warn({ workflow: parseify(newWorkflowUnsafe) }, message);
    return null;
  }

  return result.data;
};
