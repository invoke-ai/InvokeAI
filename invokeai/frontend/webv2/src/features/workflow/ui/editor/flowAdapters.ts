import type {
  FieldType,
  FieldInputTemplate,
  FieldOutputTemplate,
  InvocationTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowConnectorNode,
  WorkflowCurrentImageNode,
  WorkflowEdge,
  WorkflowInvocationNode,
  WorkflowNotesNode,
} from '@features/workflow/contracts';
import type { Edge as FlowEdge, Node as FlowNode } from '@xyflow/react';
import type { CSSProperties } from 'react';

import { isExecutableInvocationType } from '@features/workflow/graph';
import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  createWorkflowGraphIndex,
  getFieldTypeColor,
  getFieldTypeLabel,
  getResolvedWorkflowEdgesIndexed,
  getWorkflowSourceFieldType,
  getWorkflowTargetFieldType,
  type WorkflowGraphIndex,
} from '@features/workflow/utility';

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
    canUseCache: boolean;
    documentNode: WorkflowInvocationNode;
    /** Source handle names with an outgoing edge. */
    connectedSourceHandles: string[];
    /** Target handle names with an incoming edge. */
    connectedTargetHandles: string[];
    /** Field names of this node exposed in the Linear UI form. */
    exposedFieldNames: string[];
    /** Large workflows render unselected nodes without field controls until selected. */
    isCompact: boolean;
    template: InvocationNodeTemplateView | null;
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

export interface InvocationNodeTemplateView {
  hasImageOutput: boolean;
  inputTemplates: FieldInputTemplate[];
  isExecutable: boolean;
  outputTemplates: FieldOutputTemplate[];
  template: InvocationTemplate;
}

const sortByUiOrder = <T extends { uiOrder?: number | null }>(templates: T[]): T[] =>
  [...templates].sort((a, b) => (a.uiOrder ?? Number.MAX_SAFE_INTEGER) - (b.uiOrder ?? Number.MAX_SAFE_INTEGER));

const hasImageOutput = (template: InvocationTemplate): boolean =>
  template.type !== 'image' && Object.values(template.outputs).some((output) => output.type.name === 'ImageField');

const createInvocationNodeTemplateView = (template: InvocationTemplate): InvocationNodeTemplateView => ({
  hasImageOutput: hasImageOutput(template),
  inputTemplates: sortByUiOrder(Object.values(template.inputs).filter((input) => !input.uiHidden)),
  isExecutable: isExecutableInvocationType(template.type),
  outputTemplates: Object.values(template.outputs),
  template,
});

const templateViewCache = new WeakMap<InvocationTemplates, Map<string, InvocationNodeTemplateView>>();

const getInvocationNodeTemplateViews = (
  templates: InvocationTemplates | undefined
): Map<string, InvocationNodeTemplateView> => {
  if (!templates) {
    return new Map();
  }

  const cached = templateViewCache.get(templates);

  if (cached) {
    return cached;
  }

  const views = new Map<string, InvocationNodeTemplateView>();

  for (const template of Object.values(templates)) {
    views.set(template.type, createInvocationNodeTemplateView(template));
  }

  templateViewCache.set(templates, views);

  return views;
};

const getConnectedHandlesByNode = (document: ProjectGraphState, index: WorkflowGraphIndex): Map<string, string[]> => {
  const byNode = new Map<string, string[]>();

  for (const edge of getResolvedWorkflowEdgesIndexed(document.edges, index)) {
    byNode.set(edge.target, [...(byNode.get(edge.target) ?? []), edge.targetHandle]);
  }

  for (const handles of byNode.values()) {
    handles.sort();
  }

  return byNode;
};

const getConnectedSourceHandlesByNode = (document: ProjectGraphState): Map<string, string[]> => {
  const byNode = new Map<string, string[]>();

  for (const edge of document.edges) {
    byNode.set(edge.source, [...(byNode.get(edge.source) ?? []), edge.sourceHandle]);
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
  templates?: InvocationTemplates,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges),
  isCompact = false,
  canUseCache = false
): WorkflowFlowNode[] => {
  const previousById = new Map(previousNodes.map((node) => [node.id, node]));
  const connectedByNode = getConnectedHandlesByNode(document, index);
  const connectedSourcesByNode = getConnectedSourceHandlesByNode(document);
  const exposedByNode = getExposedFieldsByNode(document);
  const templateViews = getInvocationNodeTemplateViews(templates);

  return document.nodes.map((documentNode): WorkflowFlowNode => {
    const previous = previousById.get(documentNode.id);
    const selected = previous?.selected ?? false;

    if (documentNode.type === 'invocation') {
      const connectedSourceHandles = connectedSourcesByNode.get(documentNode.id) ?? EMPTY_NAMES;
      const connectedTargetHandles = connectedByNode.get(documentNode.id) ?? EMPTY_NAMES;
      const exposedFieldNames = exposedByNode.get(documentNode.id) ?? EMPTY_NAMES;
      const template = templateViews.get(documentNode.data.type) ?? null;

      if (
        previous?.type === 'invocation' &&
        previous.data.canUseCache === canUseCache &&
        previous.data.documentNode === documentNode &&
        previous.data.isCompact === isCompact &&
        previous.data.template === template &&
        sameNames(previous.data.connectedSourceHandles, connectedSourceHandles) &&
        sameNames(previous.data.connectedTargetHandles, connectedTargetHandles) &&
        sameNames(previous.data.exposedFieldNames, exposedFieldNames)
      ) {
        return previous;
      }

      return {
        data: {
          canUseCache,
          connectedSourceHandles,
          connectedTargetHandles,
          documentNode,
          exposedFieldNames,
          isCompact,
          template,
        },
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
        ? (getWorkflowTargetFieldType(document, templates, documentNode.id, CONNECTOR_INPUT_HANDLE, index) ?? null)
        : null;
      const outputFieldType = templates
        ? (getWorkflowSourceFieldType(document, templates, documentNode.id, CONNECTOR_OUTPUT_HANDLE, index) ?? null)
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
  edge: WorkflowEdge,
  index: WorkflowGraphIndex
): FieldType | null => {
  if (!templates) {
    return null;
  }

  return (
    getWorkflowSourceFieldType(document, templates, edge.source, edge.sourceHandle, index) ??
    getWorkflowTargetFieldType(document, templates, edge.target, edge.targetHandle, index) ??
    null
  );
};

export const getWorkflowEdgeData = (
  document: ProjectGraphState,
  edge: WorkflowEdge,
  pathType: FlowEdgeType,
  templates?: InvocationTemplates,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges)
): WorkflowEdgeData => {
  const fieldType = getWorkflowEdgeFieldType(document, templates, edge, index);

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
  reduceMotion = false,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges)
): WorkflowFlowEdge[] => {
  const previousById = new Map(previousEdges.map((edge) => [edge.id, edge]));

  return document.edges.map((edge) => {
    const previous = previousById.get(edge.id);
    const data = getWorkflowEdgeData(document, edge, edgeType, templates, index);
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
