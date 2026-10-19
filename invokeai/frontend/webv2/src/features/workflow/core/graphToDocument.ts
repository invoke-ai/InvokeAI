import { z } from 'zod';

import type { InvocationTemplates, ProjectGraphState, WorkflowEdge, WorkflowInvocationNode } from './types';

import {
  buildInvocationNode,
  createProjectGraph,
  createWorkflowId,
  migrateImageCollectionInputValues,
  migrateMiniMaxH3NumFramesValue,
  nodeAcceptsExtraInputs,
} from './document';
import { getLayeredPositions } from './graphLayout';
import { isInvocationNode } from './types';

/** Use a structural preview shape so core does not import UI contracts. */
export interface PreviewGraphLike {
  label?: string;
  nodes: Array<{ id: string; type: string; inputs: Record<string, unknown> }>;
  edges: Array<{
    id: string;
    sourceField: string;
    sourceNodeId: string;
    targetField: string;
    targetNodeId: string;
    type?: 'default' | 'loop_linkage';
  }>;
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
  // Lay out the full graph, including positions reserved by skipped nodes.
  const positions = getLayeredPositions(
    graph.nodes.map((node) => ({ id: node.id })),
    graph.edges.map((edge) => ({ sourceNodeId: edge.sourceNodeId, targetNodeId: edge.targetNodeId }))
  );

  const nodes: WorkflowInvocationNode[] = [];

  for (const graphNode of graph.nodes) {
    const template = Object.hasOwn(templates, graphNode.type) ? templates[graphNode.type] : undefined;

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

    for (const [key, value] of Object.entries(graphNode.inputs)) {
      if (key === 'id' || key === 'type' || key === 'is_intermediate' || key === 'use_cache') {
        continue;
      }

      if (value === undefined) {
        continue;
      }

      const instance = Object.hasOwn(node.data.inputs, key) ? node.data.inputs[key] : undefined;

      if (instance) {
        instance.value = structuredClone(value);
      } else if (nodeAcceptsExtraInputs(graphNode.type)) {
        Object.defineProperty(node.data.inputs, key, {
          configurable: true,
          enumerable: true,
          value: { label: '', name: key, value: structuredClone(value) },
          writable: true,
        });
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

    if (
      !targetTemplate ||
      (!Object.hasOwn(targetTemplate.inputs, graphEdge.targetField) && !nodeAcceptsExtraInputs(targetTemplate.type))
    ) {
      continue;
    }

    edges.push({
      id: createWorkflowId('edge'),
      source: graphEdge.sourceNodeId,
      sourceHandle: graphEdge.sourceField,
      target: graphEdge.targetNodeId,
      targetHandle: graphEdge.targetField,
      type: graphEdge.type ?? 'default',
    });
  }

  // Clear literal values on connected inputs after wiring, so a stale direct
  // value can't shadow the edge during compilation.
  for (const edge of edges) {
    const targetNode = nodes.find((node) => node.id === edge.target);
    const instance =
      targetNode && Object.hasOwn(targetNode.data.inputs, edge.targetHandle)
        ? targetNode.data.inputs[edge.targetHandle]
        : undefined;

    if (instance) {
      instance.value = undefined;
    }
  }

  const document = createProjectGraph(createWorkflowId('workflow'), graph.label || 'Untitled Workflow');

  document.nodes = nodes;
  document.edges = edges;

  return { document, skippedNodeTypes };
};

/** Images made by Generate store an execution graph instead of an authored workflow. */
export const storedGraphToDocument = (raw: unknown, templates: InvocationTemplates): PreviewGraphDocumentResult => {
  const zGraphEndpoint = z.object({ field: z.string().min(1), node_id: z.string().min(1) });
  const zStoredGraph = z.object({
    nodes: z.record(z.string(), z.looseObject({ id: z.string().min(1), type: z.string().min(1) })),
    edges: z.array(
      z.object({
        source: zGraphEndpoint,
        destination: zGraphEndpoint,
        type: z.enum(['default', 'loop_linkage']).default('default'),
      })
    ),
  });
  const graph = zStoredGraph.parse(raw);

  const result = previewGraphToDocument(
    {
      nodes: Object.values(graph.nodes).map(({ id, type, ...inputs }) => ({ id, inputs, type })),
      edges: graph.edges.map((edge, index) => ({
        id: String(index),
        sourceField: edge.source.field,
        sourceNodeId: edge.source.node_id,
        targetField: edge.destination.field,
        targetNodeId: edge.destination.node_id,
        type: edge.type,
      })),
    },
    templates
  );

  // Graphs carry no source versions. Recognize legacy values before labeling the recalled nodes current.
  // Preview conversion has already cleared connected input literals, so they cannot be mistaken for old lists.
  for (const node of result.document.nodes) {
    if (!isInvocationNode(node)) {
      continue;
    }

    const template = templates[node.data.type]!;

    if (node.data.type === 'image_collection') {
      migrateImageCollectionInputValues(node.data.inputs, template, false);
    } else if (node.data.type === 'minimax_h3_denoise') {
      migrateMiniMaxH3NumFramesValue(node.data.inputs, template);
    }
  }

  return result;
};
