import { compareVersions } from 'compare-versions';
import { cloneDeep, keyBy } from 'lodash-es';
import {
  InvocationTemplate,
  Workflow,
  WorkflowWarning,
  isWorkflowInvocationNode,
} from '../types/types';
import { parseify } from 'common/util/serialize';
import i18n from 'i18next';

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
        message: `${i18n.t('nodes.node')} "${node.data.type}" ${i18n.t(
          'nodes.skipped'
        )}`,
        issues: [
          `${i18n.t('nodes.nodeType')}"${node.data.type}" ${i18n.t(
            'nodes.doesNotExist'
          )}`,
        ],
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
        message: `${i18n.t('nodes.node')} "${node.data.type}" ${i18n.t(
          'nodes.mismatchedVersion'
        )}`,
        issues: [
          `${i18n.t('nodes.node')} "${node.data.type}" v${
            node.data.version
          } ${i18n.t('nodes.maybeIncompatible')} v${nodeTemplate.version}`,
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
      issues.push(
        `${i18n.t('nodes.outputNode')} ${edge.source} ${i18n.t(
          'nodes.doesNotExist'
        )}`
      );
    } else if (
      edge.type === 'default' &&
      !(edge.sourceHandle in sourceNode.data.outputs)
    ) {
      issues.push(
        `${i18n.t('nodes.outputNodes')} "${edge.source}.${
          edge.sourceHandle
        }" ${i18n.t('nodes.doesNotExist')}`
      );
    }
    if (!targetNode) {
      issues.push(
        `${i18n.t('nodes.inputNode')} ${edge.target} ${i18n.t(
          'nodes.doesNotExist'
        )}`
      );
    } else if (
      edge.type === 'default' &&
      !(edge.targetHandle in targetNode.data.inputs)
    ) {
      issues.push(
        `${i18n.t('nodes.inputFeilds')} "${edge.target}.${
          edge.targetHandle
        }" ${i18n.t('nodes.doesNotExist')}`
      );
    }
    if (!nodeTemplates[sourceNode?.data.type ?? '__UNKNOWN_NODE_TYPE__']) {
      issues.push(
        `${i18n.t('nodes.sourceNode')} "${edge.source}" ${i18n.t(
          'nodes.missingTemplate'
        )} "${sourceNode?.data.type}"`
      );
    }
    if (!nodeTemplates[targetNode?.data.type ?? '__UNKNOWN_NODE_TYPE__']) {
      issues.push(
        `${i18n.t('nodes.sourceNode')}"${edge.target}" ${i18n.t(
          'nodes.missingTemplate'
        )} "${targetNode?.data.type}"`
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
