import { logger } from 'app/logging/logger';
import { deepClone } from 'common/util/deepClone';
import { parseify } from 'common/util/serialize';
import type { NodesState, WorkflowsState } from 'features/nodes/store/types';
import { isInvocationNode, isNotesNode } from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { zWorkflowV3 } from 'features/nodes/types/workflow';
import i18n from 'i18n';
import { pick } from 'lodash-es';
import { fromZodError } from 'zod-validation-error';

const log = logger('workflows');

export type BuildWorkflowArg = {
  nodes: NodesState['nodes'];
  edges: NodesState['edges'];
  workflow: WorkflowsState;
};

const workflowKeys = [
  'name',
  'author',
  'description',
  'version',
  'contact',
  'tags',
  'notes',
  'exposedFields',
  'meta',
  'id',
  'form',
] satisfies (keyof WorkflowV3)[];

type BuildWorkflowFunction = (arg: BuildWorkflowArg) => WorkflowV3;

export const buildWorkflowFast: BuildWorkflowFunction = ({ nodes, edges, workflow }: BuildWorkflowArg): WorkflowV3 => {
  const clonedWorkflow = pick(deepClone(workflow), workflowKeys);

  const newWorkflow: WorkflowV3 = {
    ...clonedWorkflow,
    nodes: [],
    edges: [],
  };

  nodes.forEach((node) => {
    if (isInvocationNode(node) && node.type) {
      newWorkflow.nodes.push({
        id: node.id,
        type: node.type,
        data: deepClone(node.data),
        position: { ...node.position },
      });
    } else if (isNotesNode(node) && node.type) {
      newWorkflow.nodes.push({
        id: node.id,
        type: node.type,
        data: deepClone(node.data),
        position: { ...node.position },
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
        hidden: edge.hidden,
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

export const buildWorkflowWithValidation = ({ nodes, edges, workflow }: BuildWorkflowArg): WorkflowV3 | null => {
  // builds what really, really should be a valid workflow
  const workflowToValidate = buildWorkflowFast({ nodes, edges, workflow });

  // but bc we are storing this in the DB, let's be extra sure
  const result = zWorkflowV3.safeParse(workflowToValidate);

  if (!result.success) {
    const { message } = fromZodError(result.error, {
      prefix: i18n.t('nodes.unableToValidateWorkflow'),
    });

    log.warn({ workflow: parseify(workflowToValidate) }, message);
    return null;
  }

  return result.data;
};
