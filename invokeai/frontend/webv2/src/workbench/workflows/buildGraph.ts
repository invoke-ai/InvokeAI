import type { BackendGraphContract, GraphContract } from '@workbench/types';

import type { InvocationTemplatesSnapshot } from './templates';
import type { FieldInputTemplate, InvocationTemplates, ProjectGraphState, WorkflowInvocationNode } from './types';

import { getResolvedWorkflowEdges } from './connectors';
import { createWorkflowId } from './document';
import { isInvocationNode } from './types';
import { hasAnyCycle } from './validation';

/**
 * Compiles the project graph document into the immutable, queue-facing
 * `GraphContract`. Ported from the legacy `buildNodesGraph`, with connector
 * resolution and without batch handling (batch/generator nodes are rejected by
 * readiness until batching lands).
 */

/** Client-resolved batch/generator nodes from the legacy editor; executing them server-side is meaningless. */
const UNSUPPORTED_NODE_TYPES = new Set([
  'float_batch',
  'float_generator',
  'image_batch',
  'image_generator',
  'integer_batch',
  'integer_generator',
  'string_batch',
  'string_generator',
]);

const getExecutableNodes = (document: ProjectGraphState): WorkflowInvocationNode[] =>
  document.nodes.filter(isInvocationNode);

const isEmptyValue = (value: unknown): boolean =>
  value === undefined || value === null || (typeof value === 'string' && value.trim() === '');

const getNodeDisplayName = (node: WorkflowInvocationNode, templates: InvocationTemplates): string =>
  node.data.label || templates[node.data.type]?.title || node.data.type;

/**
 * Translates a board field value to the backend shape: `auto` and `none`
 * sentinels are omitted so the backend applies its default board behavior.
 */
const toBoardGraphValue = (value: unknown): unknown => {
  if (value === 'auto' || value === 'none' || isEmptyValue(value)) {
    return undefined;
  }

  return value;
};

export interface ProjectGraphReadiness {
  canInvoke: boolean;
  reasons: string[];
}

export const getProjectGraphReadiness = (
  document: ProjectGraphState,
  templatesSnapshot: InvocationTemplatesSnapshot
): ProjectGraphReadiness => {
  if (templatesSnapshot.status === 'error') {
    return { canInvoke: false, reasons: ['Node definitions failed to load from the backend.'] };
  }

  if (templatesSnapshot.status !== 'loaded') {
    return { canInvoke: false, reasons: ['Node definitions are still loading.'] };
  }

  const templates = templatesSnapshot.templates;
  const executableNodes = getExecutableNodes(document);

  if (executableNodes.length === 0) {
    return { canInvoke: false, reasons: ['The project graph has no nodes. Add nodes in the Workflow view.'] };
  }

  const reasons: string[] = [];
  const connectedInputs = new Set(
    getResolvedWorkflowEdges(document.nodes, document.edges, templates)
      .filter((edge) => executableNodes.some((node) => node.id === edge.target))
      .map((edge) => `${edge.target}:${edge.targetHandle}`)
  );

  for (const node of executableNodes) {
    const template = templates[node.data.type];

    if (!template) {
      reasons.push(`Unknown node type "${node.data.type}".`);
      continue;
    }

    if (UNSUPPORTED_NODE_TYPES.has(node.data.type)) {
      reasons.push(`Batch/generator node "${getNodeDisplayName(node, templates)}" is not supported yet.`);
      continue;
    }

    for (const inputTemplate of Object.values(template.inputs)) {
      if (!inputTemplate.required) {
        continue;
      }

      if (connectedInputs.has(`${node.id}:${inputTemplate.name}`)) {
        continue;
      }

      if (inputTemplate.input === 'connection') {
        reasons.push(`"${getNodeDisplayName(node, templates)}" is missing a connection for "${inputTemplate.title}".`);
        continue;
      }

      if (isEmptyValue(node.data.inputs[inputTemplate.name]?.value)) {
        reasons.push(`"${getNodeDisplayName(node, templates)}" is missing required input "${inputTemplate.title}".`);
      }
    }
  }

  if (hasAnyCycle(document.nodes, document.edges)) {
    reasons.push('The project graph contains a cycle.');
  }

  return { canInvoke: reasons.length === 0, reasons };
};

const toGraphInputValue = (inputTemplate: FieldInputTemplate, value: unknown): unknown => {
  if (inputTemplate.type.name === 'BoardField') {
    return toBoardGraphValue(value);
  }

  return value;
};

/** Compiles the document into a `GraphContract` carrying the executable backend graph. */
export const compileProjectGraph = (document: ProjectGraphState, templates: InvocationTemplates): GraphContract => {
  const executableNodes = getExecutableNodes(document).filter((node) => templates[node.data.type] !== undefined);
  const executableNodeIds = new Set(executableNodes.map((node) => node.id));
  const backendGraph: BackendGraphContract = { edges: [], id: createWorkflowId('workflow-graph'), nodes: {} };

  for (const node of executableNodes) {
    const template = templates[node.data.type] as NonNullable<(typeof templates)[string]>;
    const graphNode: Record<string, unknown> = {
      id: node.id,
      is_intermediate: node.data.isIntermediate,
      type: node.data.type,
      use_cache: node.data.useCache,
    };

    for (const instance of Object.values(node.data.inputs)) {
      const inputTemplate = template.inputs[instance.name];

      if (!inputTemplate || instance.value === undefined) {
        continue;
      }

      const value = toGraphInputValue(inputTemplate, instance.value);

      if (value !== undefined) {
        graphNode[instance.name] = value;
      }
    }

    backendGraph.nodes[node.id] = graphNode as BackendGraphContract['nodes'][string];
  }

  const seenEdgeKeys = new Set<string>();

  for (const edge of getResolvedWorkflowEdges(document.nodes, document.edges, templates)) {
    if (!executableNodeIds.has(edge.source) || !executableNodeIds.has(edge.target)) {
      continue;
    }

    const key = `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;

    if (seenEdgeKeys.has(key)) {
      continue;
    }

    seenEdgeKeys.add(key);
    backendGraph.edges.push({
      destination: { field: edge.targetHandle, node_id: edge.target },
      source: { field: edge.sourceHandle, node_id: edge.source },
    });

    // A connected input always wins over a stale direct value; sending both
    // would let pydantic reject the node on the ignored direct value.
    const targetNode = backendGraph.nodes[edge.target];

    if (targetNode) {
      delete targetNode[edge.targetHandle];
    }
  }

  return {
    backendGraph,
    edges: getResolvedWorkflowEdges(document.nodes, document.edges, templates)
      .filter((edge) => executableNodeIds.has(edge.source) && executableNodeIds.has(edge.target))
      .map((edge) => ({
        id: edge.id,
        sourceField: edge.sourceHandle,
        sourceNodeId: edge.source,
        targetField: edge.targetHandle,
        targetNodeId: edge.target,
      })),
    id: backendGraph.id,
    label: document.name || 'Workflow',
    nodes: executableNodes.map((node) => ({
      id: node.id,
      inputs: Object.fromEntries(Object.values(node.data.inputs).map((instance) => [instance.name, instance.value])),
      type: node.data.type,
    })),
    updatedAt: new Date().toISOString(),
    version: 1,
  };
};
