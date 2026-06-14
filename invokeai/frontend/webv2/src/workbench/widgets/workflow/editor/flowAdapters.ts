import type { Edge as FlowEdge, Node as FlowNode } from '@xyflow/react';

import type {
  ProjectGraphState,
  WorkflowCurrentImageNode,
  WorkflowInvocationNode,
  WorkflowNotesNode,
} from '../../../workflows/types';

/**
 * Adapters between the project graph document and xyflow's node/edge state.
 * The document is the source of truth; xyflow state is rebuilt from it on
 * every document change, carrying over transient view state (selection).
 *
 * Rebuilds preserve object identity for unchanged nodes/edges so memoized
 * node components skip re-rendering (see React Flow's performance guidance) —
 * the document reducer already keeps untouched node identities stable.
 * Derived per-node facts the node components need (incoming connections,
 * Linear-UI exposure) are precomputed into `data` here, so node components
 * never subscribe to workbench state.
 */

export type InvocationFlowNode = FlowNode<
  {
    documentNode: WorkflowInvocationNode;
    /** Target handle names with an incoming edge. */
    connectedTargetHandles: string[];
    /** Field names of this node exposed in the Linear UI form. */
    exposedFieldNames: string[];
  },
  'invocation'
>;
export type NotesFlowNode = FlowNode<{ documentNode: WorkflowNotesNode }, 'notes'>;
export type CurrentImageFlowNode = FlowNode<{ documentNode: WorkflowCurrentImageNode }, 'current_image'>;
export type WorkflowFlowNode = InvocationFlowNode | NotesFlowNode | CurrentImageFlowNode;

const EMPTY_NAMES: string[] = [];

const sameNames = (a: string[], b: string[]): boolean => a.length === b.length && a.every((name, i) => name === b[i]);

const getConnectedHandlesByNode = (document: ProjectGraphState): Map<string, string[]> => {
  const byNode = new Map<string, string[]>();

  for (const edge of document.edges) {
    byNode.set(edge.target, [...(byNode.get(edge.target) ?? []), edge.targetHandle]);
  }

  for (const handles of byNode.values()) {
    handles.sort();
  }

  return byNode;
};

const getExposedFieldsByNode = (document: ProjectGraphState): Map<string, string[]> => {
  const byNode = new Map<string, string[]>();

  for (const element of Object.values(document.form.elements)) {
    if (element.type === 'node-field') {
      const { fieldName, nodeId } = element.data.fieldIdentifier;

      byNode.set(nodeId, [...(byNode.get(nodeId) ?? []), fieldName]);
    }
  }

  for (const fields of byNode.values()) {
    fields.sort();
  }

  return byNode;
};

export const toFlowNodes = (
  document: ProjectGraphState,
  previousNodes: WorkflowFlowNode[] = []
): WorkflowFlowNode[] => {
  const previousById = new Map(previousNodes.map((node) => [node.id, node]));
  const connectedByNode = getConnectedHandlesByNode(document);
  const exposedByNode = getExposedFieldsByNode(document);

  return document.nodes.map((documentNode): WorkflowFlowNode => {
    const previous = previousById.get(documentNode.id);
    const selected = previous?.selected ?? false;

    if (documentNode.type === 'invocation') {
      const connectedTargetHandles = connectedByNode.get(documentNode.id) ?? EMPTY_NAMES;
      const exposedFieldNames = exposedByNode.get(documentNode.id) ?? EMPTY_NAMES;

      if (
        previous?.type === 'invocation' &&
        previous.data.documentNode === documentNode &&
        sameNames(previous.data.connectedTargetHandles, connectedTargetHandles) &&
        sameNames(previous.data.exposedFieldNames, exposedFieldNames)
      ) {
        return previous;
      }

      return {
        data: { connectedTargetHandles, documentNode, exposedFieldNames },
        id: documentNode.id,
        position: documentNode.position,
        selected,
        type: 'invocation' as const,
      };
    }

    if (documentNode.type === 'notes') {
      if (previous?.type === 'notes' && previous.data.documentNode === documentNode) {
        return previous;
      }

      return {
        data: { documentNode },
        id: documentNode.id,
        position: documentNode.position,
        selected,
        type: 'notes' as const,
      };
    }

    if (previous?.type === 'current_image' && previous.data.documentNode === documentNode) {
      return previous;
    }

    return {
      data: { documentNode },
      id: documentNode.id,
      position: documentNode.position,
      selected,
      type: 'current_image' as const,
    };
  });
};

/** Returns nodes with exactly `selectedIds` selected, preserving identity where unchanged. */
export const withNodeSelection = (nodes: WorkflowFlowNode[], selectedIds: Set<string>): WorkflowFlowNode[] =>
  nodes.map((node) => {
    const selected = selectedIds.has(node.id);

    if ((node.selected ?? false) === selected) {
      return node;
    }

    // Narrowed per branch so TS keeps the node type / data correlation through the spread.
    if (node.type === 'invocation') {
      return { ...node, selected };
    }

    if (node.type === 'notes') {
      return { ...node, selected };
    }

    return { ...node, selected };
  });

/** xyflow edge component per the user's connection-style preference. */
export type FlowEdgeType = 'default' | 'straight';

export const toFlowEdges = (
  document: ProjectGraphState,
  previousEdges: FlowEdge[] = [],
  edgeType: FlowEdgeType = 'default'
): FlowEdge[] => {
  const previousById = new Map(previousEdges.map((edge) => [edge.id, edge]));

  return document.edges.map((edge) => {
    const previous = previousById.get(edge.id);

    if (
      previous &&
      previous.type === edgeType &&
      previous.source === edge.source &&
      previous.sourceHandle === edge.sourceHandle &&
      previous.target === edge.target &&
      previous.targetHandle === edge.targetHandle
    ) {
      return previous;
    }

    return {
      id: edge.id,
      selected: previous?.selected ?? false,
      source: edge.source,
      sourceHandle: edge.sourceHandle,
      target: edge.target,
      targetHandle: edge.targetHandle,
      type: edgeType,
    };
  });
};
