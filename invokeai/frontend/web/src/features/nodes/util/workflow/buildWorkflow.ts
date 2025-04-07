import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { parseify } from 'common/util/serialize';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState } from 'features/nodes/store/types';
import { isInvocationNode, isNotesNode } from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { zWorkflowV3 } from 'features/nodes/types/workflow';
import i18n from 'i18n';
import { pick } from 'lodash-es';
import { useCallback } from 'react';
import { fromZodError } from 'zod-validation-error';

const log = logger('workflows');

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

export const buildWorkflowFast = (nodesState: NodesState): WorkflowV3 => {
  const { nodes, edges, ...rest } = nodesState;
  const clonedWorkflow = pick(rest, workflowKeys);

  const newWorkflow: WorkflowV3 = {
    ...clonedWorkflow,
    nodes: [],
    edges: [],
  };

  for (const node of nodes) {
    if (isInvocationNode(node) && node.type) {
      const { id, type, data, position } = node;
      newWorkflow.nodes.push({ id, type, data, position });
    } else if (isNotesNode(node) && node.type) {
      const { id, type, data, position } = node;
      newWorkflow.nodes.push({ id, type, data, position });
    }
  }

  for (const edge of edges) {
    if (edge.type === 'default' && edge.sourceHandle && edge.targetHandle) {
      const { id, type, source, target, sourceHandle, targetHandle, hidden } = edge;
      newWorkflow.edges.push({ id, type, source, target, sourceHandle, targetHandle, hidden });
    } else if (edge.type === 'collapsed') {
      const { id, type, source, target } = edge;
      newWorkflow.edges.push({ id, type, source, target });
    }
  }

  return deepClone(newWorkflow);
};

export const buildWorkflowWithValidation = (nodesState: NodesState): WorkflowV3 | null => {
  // builds what really, really should be a valid workflow
  const workflowToValidate = buildWorkflowFast(nodesState);

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

export const useBuildWorkflowFast = (): (() => WorkflowV3) => {
  const store = useAppStore();
  const buildWorkflow = useCallback(() => {
    const nodesState = selectNodesSlice(store.getState());
    return buildWorkflowFast(nodesState);
  }, [store]);

  return buildWorkflow;
};
