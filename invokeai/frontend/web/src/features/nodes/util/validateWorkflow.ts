import { compareVersions } from 'compare-versions';
import { cloneDeep, keyBy } from 'lodash-es';
import {
  InvocationTemplate,
  Workflow,
  WorkflowWarning,
  isWorkflowInvocationNode,
} from '../types/types';
import { parseify } from 'common/util/serialize';

export const validateWorkflow = (
  workflow: Workflow,
  nodeTemplates: Record<string, InvocationTemplate>
) => {
  const clone = cloneDeep(workflow);
  const { nodes, edges } = clone;
  const errors: WorkflowWarning[] = [];
  const invocationNodes = nodes.filter(isWorkflowInvocationNode);
  const keyedNodes = keyBy(invocationNodes, 'id');
  nodes.forEach((node) => {
    if (!isWorkflowInvocationNode(node)) {
      return;
    }

    const nodeTemplate = nodeTemplates[node.data.type];
    if (!nodeTemplate) {
      errors.push({
        message: `Node "${node.data.type}" skipped`,
        issues: [`Node type "${node.data.type}" does not exist`],
        data: node,
      });
      return;
    }

    if (
      nodeTemplate.version &&
      node.data.version &&
      compareVersions(nodeTemplate.version, node.data.version) !== 0
    ) {
      errors.push({
        message: `Node "${node.data.type}" has mismatched version`,
        issues: [
          `Node "${node.data.type}" v${node.data.version} may be incompatible with installed v${nodeTemplate.version}`,
        ],
        data: { node, nodeTemplate: parseify(nodeTemplate) },
      });
      return;
    }
  });
  edges.forEach((edge, i) => {
    const sourceNode = keyedNodes[edge.source];
    const targetNode = keyedNodes[edge.target];
    const issues: string[] = [];
    if (!sourceNode) {
      issues.push(`Output node ${edge.source} does not exist`);
    } else if (
      edge.type === 'default' &&
      !(edge.sourceHandle in sourceNode.data.outputs)
    ) {
      issues.push(
        `Output field "${edge.source}.${edge.sourceHandle}" does not exist`
      );
    }
    if (!targetNode) {
      issues.push(`Input node ${edge.target} does not exist`);
    } else if (
      edge.type === 'default' &&
      !(edge.targetHandle in targetNode.data.inputs)
    ) {
      issues.push(
        `Input field "${edge.target}.${edge.targetHandle}" does not exist`
      );
    }
    if (!nodeTemplates[sourceNode?.data.type ?? '__UNKNOWN_NODE_TYPE__']) {
      issues.push(
        `Source node "${edge.source}" missing template "${sourceNode?.data.type}"`
      );
    }
    if (!nodeTemplates[targetNode?.data.type ?? '__UNKNOWN_NODE_TYPE__']) {
      issues.push(
        `Source node "${edge.target}" missing template "${targetNode?.data.type}"`
      );
    }
    if (issues.length) {
      delete edges[i];
      const src = edge.type === 'default' ? edge.sourceHandle : edge.source;
      const tgt = edge.type === 'default' ? edge.targetHandle : edge.target;
      errors.push({
        message: `Edge "${src} -> ${tgt}" skipped`,
        issues,
        data: edge,
      });
    }
  });
  return { workflow: clone, errors };
};
