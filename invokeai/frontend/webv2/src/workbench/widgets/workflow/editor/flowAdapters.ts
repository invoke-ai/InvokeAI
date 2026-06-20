import type {
  FieldType,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowConnectorNode,
  WorkflowCurrentImageNode,
  WorkflowEdge,
  WorkflowInvocationNode,
  WorkflowNotesNode,
} from '@workbench/workflows/types';
import type { Edge as FlowEdge, Node as FlowNode } from '@xyflow/react';
import type { CSSProperties } from 'react';

import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  getResolvedWorkflowEdges,
} from '@workbench/workflows/connectors';
import { getFieldTypeColor, getFieldTypeLabel } from '@workbench/workflows/fields';
import { getWorkflowSourceFieldType, getWorkflowTargetFieldType } from '@workbench/workflows/validation';

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
export type ConnectorFlowNode = FlowNode<
  { documentNode: WorkflowConnectorNode; inputFieldType: FieldType | null; outputFieldType: FieldType | null },
  'connector'
>;
export type WorkflowFlowNode = InvocationFlowNode | NotesFlowNode | CurrentImageFlowNode | ConnectorFlowNode;

const EMPTY_NAMES: string[] = [];
const EMPTY_NODE_IDS = new Set<string>();
const SELECTED_NODE_EDGE_CLASS = 'workflow-selected-node-edge';
const SELECTED_NODE_EDGE_STYLE: CSSProperties = { strokeWidth: 2 };

const sameNames = (a: string[], b: string[]): boolean => a.length === b.length && a.every((name, i) => name === b[i]);

const getConnectedHandlesByNode = (document: ProjectGraphState): Map<string, string[]> => {
  const byNode = new Map<string, string[]>();

  for (const edge of getResolvedWorkflowEdges(document.nodes, document.edges)) {
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
  previousNodes: WorkflowFlowNode[] = [],
  templates?: InvocationTemplates
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

    if (documentNode.type === 'connector') {
      const inputFieldType = templates
        ? (getWorkflowTargetFieldType(document, templates, documentNode.id, CONNECTOR_INPUT_HANDLE) ?? null)
        : null;
      const outputFieldType = templates
        ? (getWorkflowSourceFieldType(document, templates, documentNode.id, CONNECTOR_OUTPUT_HANDLE) ?? null)
        : null;

      if (
        previous?.type === 'connector' &&
        previous.data.documentNode === documentNode &&
        previous.data.inputFieldType === inputFieldType &&
        previous.data.outputFieldType === outputFieldType
      ) {
        return previous;
      }

      return {
        data: { documentNode, inputFieldType, outputFieldType },
        id: documentNode.id,
        position: documentNode.position,
        selected,
        type: 'connector' as const,
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

    if (node.type === 'connector') {
      return { ...node, selected };
    }

    return { ...node, selected };
  });

/** xyflow edge component per the user's connection-style preference. */
export type FlowEdgeType = 'default' | 'step';

export interface WorkflowEdgeData extends Record<string, unknown> {
  fieldTypeLabel: string;
  pathType: FlowEdgeType;
  stroke: string;
  strokeDasharray?: string;
  strokeWidth: number;
  tooltip: string;
}

export type WorkflowFlowEdge = FlowEdge<WorkflowEdgeData, FlowEdgeType>;

const UNKNOWN_EDGE_DATA = (pathType: FlowEdgeType): WorkflowEdgeData => ({
  fieldTypeLabel: 'Unknown',
  pathType,
  stroke: 'var(--xy-edge-stroke)',
  strokeWidth: 2,
  tooltip: 'Unknown field type',
});

const getWorkflowEdgeFieldType = (
  document: ProjectGraphState,
  templates: InvocationTemplates | undefined,
  edge: WorkflowEdge
): FieldType | null => {
  if (!templates) {
    return null;
  }

  return (
    getWorkflowSourceFieldType(document, templates, edge.source, edge.sourceHandle) ??
    getWorkflowTargetFieldType(document, templates, edge.target, edge.targetHandle) ??
    null
  );
};

export const getWorkflowEdgeData = (
  document: ProjectGraphState,
  edge: WorkflowEdge,
  pathType: FlowEdgeType,
  templates?: InvocationTemplates
): WorkflowEdgeData => {
  const fieldType = getWorkflowEdgeFieldType(document, templates, edge);

  if (!fieldType) {
    return UNKNOWN_EDGE_DATA(pathType);
  }

  const fieldTypeLabel = getFieldTypeLabel(fieldType);
  const strokeDasharray = fieldType.batch
    ? '2 5'
    : fieldType.cardinality === 'COLLECTION'
      ? '8 4'
      : fieldType.cardinality === 'SINGLE_OR_COLLECTION'
        ? '8 3 2 3'
        : undefined;

  return {
    fieldTypeLabel,
    pathType,
    stroke: getFieldTypeColor(fieldType),
    strokeDasharray,
    strokeWidth: fieldType.cardinality === 'SINGLE' && !fieldType.batch ? 2 : 2.5,
    tooltip: fieldType.batch ? `${fieldTypeLabel} batch` : fieldTypeLabel,
  };
};

const isSameEdgeData = (a: WorkflowEdgeData | undefined, b: WorkflowEdgeData): boolean =>
  a?.fieldTypeLabel === b.fieldTypeLabel &&
  a.pathType === b.pathType &&
  a.stroke === b.stroke &&
  a.strokeDasharray === b.strokeDasharray &&
  a.strokeWidth === b.strokeWidth &&
  a.tooltip === b.tooltip;

export const toFlowEdges = (
  document: ProjectGraphState,
  previousEdges: WorkflowFlowEdge[] = [],
  edgeType: FlowEdgeType = 'default',
  selectedNodeIds: Set<string> = EMPTY_NODE_IDS,
  templates?: InvocationTemplates,
  reduceMotion = false
): WorkflowFlowEdge[] => {
  const previousById = new Map(previousEdges.map((edge) => [edge.id, edge]));

  return document.edges.map((edge) => {
    const previous = previousById.get(edge.id);
    const data = getWorkflowEdgeData(document, edge, edgeType, templates);
    const isConnectedToSelectedNode = selectedNodeIds.has(edge.source) || selectedNodeIds.has(edge.target);
    const animated = isConnectedToSelectedNode && !reduceMotion ? true : undefined;
    const className = isConnectedToSelectedNode ? SELECTED_NODE_EDGE_CLASS : undefined;
    const style = isConnectedToSelectedNode ? SELECTED_NODE_EDGE_STYLE : undefined;
    const zIndex = isConnectedToSelectedNode ? 1000 : undefined;

    if (
      previous &&
      previous.type === edgeType &&
      previous.source === edge.source &&
      previous.sourceHandle === edge.sourceHandle &&
      previous.target === edge.target &&
      previous.targetHandle === edge.targetHandle &&
      previous.animated === animated &&
      previous.className === className &&
      isSameEdgeData(previous.data, data) &&
      previous.style === style &&
      previous.zIndex === zIndex
    ) {
      return previous;
    }

    return {
      id: edge.id,
      animated,
      className,
      data,
      selected: previous?.selected ?? false,
      source: edge.source,
      sourceHandle: edge.sourceHandle,
      style,
      target: edge.target,
      targetHandle: edge.targetHandle,
      type: edgeType,
      zIndex,
    };
  });
};
