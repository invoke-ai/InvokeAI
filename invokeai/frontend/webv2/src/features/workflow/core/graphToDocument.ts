import type { InvocationTemplates, ProjectGraphState, WorkflowEdge, WorkflowInvocationNode } from './types';

import { buildInvocationNode, createProjectGraph, createWorkflowId } from './document';
import { getLayeredPositions } from './graphLayout';

/**
 * Structural shape of the read-only preview graph (Task 1's contract graph,
 * as rendered by `GraphPreviewFlow.tsx`). Kept structural rather than
 * importing the ui contract so this module stays a pure `core` dependency.
 */
export interface PreviewGraphLike {
  label?: string;
  nodes: Array<{ id: string; type: string; inputs: Record<string, unknown> }>;
  edges: Array<{ id: string; sourceField: string; sourceNodeId: string; targetField: string; targetNodeId: string }>;
}

export interface PreviewGraphDocumentResult {
  document: ProjectGraphState;
  skippedNodeTypes: string[];
}

/** Converts a read-only preview graph into an editable project graph document. */
export const previewGraphToDocument = (
  graph: PreviewGraphLike,
  templates: InvocationTemplates
): PreviewGraphDocumentResult => {
  const skippedNodeTypes: string[] = [];
  const seenSkippedTypes = new Set<string>();
  // Positions are derived from the full graph (skipped nodes still occupy a
  // depth/row), matching Task 6's documented behavior rather than inventing
  // a hint mechanism to exclude them.
  const positions = getLayeredPositions(
    graph.nodes.map((node) => ({ id: node.id })),
    graph.edges.map((edge) => ({ sourceNodeId: edge.sourceNodeId, targetNodeId: edge.targetNodeId }))
  );

  const nodes: WorkflowInvocationNode[] = [];

  for (const graphNode of graph.nodes) {
    const template = templates[graphNode.type];

    if (!template) {
      if (!seenSkippedTypes.has(graphNode.type)) {
        seenSkippedTypes.add(graphNode.type);
        skippedNodeTypes.push(graphNode.type);
      }
      continue;
    }

    const node = buildInvocationNode(template, positions[graphNode.id] ?? { x: 0, y: 0 });

    node.id = graphNode.id;

    const isIntermediate = graphNode.inputs.is_intermediate;

    if (typeof isIntermediate === 'boolean') {
      node.data.isIntermediate = isIntermediate;
    }

    const useCache = graphNode.inputs.use_cache;

    if (typeof useCache === 'boolean') {
      node.data.useCache = useCache;
    }

    for (const key of Object.keys(template.inputs)) {
      const value = graphNode.inputs[key];

      if (value === undefined) {
        continue;
      }

      const instance = node.data.inputs[key];

      if (instance) {
        instance.value = structuredClone(value);
      }
    }

    nodes.push(node);
  }

  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges: WorkflowEdge[] = [];

  for (const graphEdge of graph.edges) {
    if (!nodeIds.has(graphEdge.sourceNodeId) || !nodeIds.has(graphEdge.targetNodeId)) {
      continue;
    }

    const targetNode = nodes.find((node) => node.id === graphEdge.targetNodeId);
    const targetTemplate = targetNode ? templates[targetNode.data.type] : undefined;

    if (!targetTemplate || !(graphEdge.targetField in targetTemplate.inputs)) {
      continue;
    }

    edges.push({
      id: createWorkflowId('edge'),
      source: graphEdge.sourceNodeId,
      sourceHandle: graphEdge.sourceField,
      target: graphEdge.targetNodeId,
      targetHandle: graphEdge.targetField,
      type: 'default',
    });
  }

  // Clear literal values on connected inputs after wiring, so a stale direct
  // value can't shadow the edge during compilation.
  for (const edge of edges) {
    const targetNode = nodes.find((node) => node.id === edge.target);
    const instance = targetNode?.data.inputs[edge.targetHandle];

    if (instance) {
      instance.value = undefined;
    }
  }

  const document = createProjectGraph(createWorkflowId('workflow'), graph.label || 'Untitled Workflow');

  document.nodes = nodes;
  document.edges = edges;

  return { document, skippedNodeTypes };
};
