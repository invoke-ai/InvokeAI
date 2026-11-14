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
  WorkflowNode,
  WorkflowNotesNode,
} from '@features/workflow/contracts';
import type { Edge as FlowEdge, Node as FlowNode } from '@xyflow/react';
import type { CSSProperties } from 'react';

import { isExecutableInvocationType } from '@features/workflow/graph';
import { LOOP_LINKAGE_STROKE } from '@features/workflow/ui/loopLinkage';
import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  createWorkflowGraphIndex,
  getFieldTypeColor,
  getFieldTypeLabel,
  getEdgesWithLoopLinkageAliases,
  getResolvedWorkflowEdgesIndexed,
  getWorkflowSourceFieldType,
  getWorkflowTargetFieldType,
  type WorkflowGraphIndex,
} from '@features/workflow/utility';

/**
 * Rebuild from document truth while preserving unchanged identities and selection. Precompute connection/exposure
 * facts so nodes need no workbench subscriptions.
 */

/**
 * Whether node definitions are known yet. A node without a template is unknown only once they have loaded; until then
 * it is waiting, and after a failed load nothing can be said about its type.
 */
export type InvocationTemplateStatus = 'error' | 'loaded' | 'loading';

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
    templateStatus: InvocationTemplateStatus;
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

const dynamicTemplateViewCache = new WeakMap<
  InvocationTemplate,
  WeakMap<NonNullable<WorkflowInvocationNode['data']['dynamicInputTemplates']>, InvocationNodeTemplateView>
>();

const getInvocationNodeTemplateView = (
  template: InvocationNodeTemplateView,
  dynamicInputTemplates: WorkflowInvocationNode['data']['dynamicInputTemplates']
): InvocationNodeTemplateView => {
  if (!dynamicInputTemplates || Object.keys(dynamicInputTemplates).length === 0) {
    return template;
  }

  let cached = dynamicTemplateViewCache.get(template.template);

  if (!cached) {
    cached = new WeakMap();
    dynamicTemplateViewCache.set(template.template, cached);
  }

  const existing = cached.get(dynamicInputTemplates);

  if (existing) {
    return existing;
  }

  const view = createInvocationNodeTemplateView({
    ...template.template,
    inputs: { ...template.template.inputs, ...dynamicInputTemplates },
  });
  cached.set(dynamicInputTemplates, view);

  return view;
};

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
  canUseCache = false,
  templateStatus: InvocationTemplateStatus = 'loaded'
): WorkflowFlowNode[] => {
  const previousById = new Map(previousNodes.map((node) => [node.id, node]));
  const connectedByNode = getConnectedHandlesByNode(document, index);
  const connectedSourcesByNode = getConnectedSourceHandlesByNode(document);
  const exposedByNode = getExposedFieldsByNode(document);
  const templateViews = getInvocationNodeTemplateViews(templates);

  return document.nodes.map((documentNode): WorkflowFlowNode => {
    const previous = previousById.get(documentNode.id);
    const selected = previous?.selected ?? false;
    // Rebuilt nodes keep their measured size so xyflow does not treat them as unmeasured (0x0) until remeasured.
    const measured = previous?.measured;

    if (documentNode.type === 'invocation') {
      const connectedSourceHandles = connectedSourcesByNode.get(documentNode.id) ?? EMPTY_NAMES;
      const connectedTargetHandles = connectedByNode.get(documentNode.id) ?? EMPTY_NAMES;
      const exposedFieldNames = exposedByNode.get(documentNode.id) ?? EMPTY_NAMES;
      const baseTemplate = templateViews.get(documentNode.data.type);
      const template = baseTemplate
        ? getInvocationNodeTemplateView(baseTemplate, documentNode.data.dynamicInputTemplates)
        : null;

      if (
        previous?.type === 'invocation' &&
        previous.data.canUseCache === canUseCache &&
        previous.data.documentNode === documentNode &&
        previous.data.isCompact === isCompact &&
        previous.data.template === template &&
        previous.data.templateStatus === templateStatus &&
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
          templateStatus,
        },
        id: documentNode.id,
        measured,
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
        measured,
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
        measured,
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
      measured,
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
export type WorkflowFlowEdgeType = FlowEdgeType | 'loop_linkage';

/** The edge component names the edge from these facts, so the copy follows the UI language. */
export interface WorkflowEdgeData extends Record<string, unknown> {
  /**
   * Set on the one edge drawn between two collapsed nodes: every document edge it stands for, itself first. The rest
   * are hidden, and selecting or deleting the drawn edge acts on all of them.
   */
  collapsedEdgeIds?: string[];
  /** The carried field type's name; null for a loop linkage or a type no template declares. */
  fieldTypeLabel: string | null;
  pathType: FlowEdgeType;
  isBatch?: boolean;
  isLoopLinkage?: boolean;
  stroke: string;
  strokeDasharray?: string;
  strokeWidth: number;
}

export type WorkflowFlowEdge = FlowEdge<WorkflowEdgeData, WorkflowFlowEdgeType>;

const UNKNOWN_EDGE_DATA = (pathType: FlowEdgeType): WorkflowEdgeData => ({
  fieldTypeLabel: null,
  pathType,
  stroke: 'var(--xy-edge-stroke)',
  strokeWidth: 2,
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
  if (edge.type === 'loop_linkage') {
    return {
      fieldTypeLabel: null,
      isLoopLinkage: true,
      pathType,
      stroke: LOOP_LINKAGE_STROKE,
      strokeDasharray: '6 4',
      strokeWidth: 2,
    };
  }

  const fieldType = getWorkflowEdgeFieldType(document, templates, edge, index);

  if (!fieldType) {
    return UNKNOWN_EDGE_DATA(pathType);
  }

  const strokeDasharray = fieldType.batch
    ? '2 5'
    : fieldType.cardinality === 'COLLECTION'
      ? '8 4'
      : fieldType.cardinality === 'SINGLE_OR_COLLECTION'
        ? '8 3 2 3'
        : undefined;

  return {
    fieldTypeLabel: getFieldTypeLabel(fieldType),
    isBatch: fieldType.batch,
    pathType,
    stroke: getFieldTypeColor(fieldType),
    strokeDasharray,
    strokeWidth: fieldType.cardinality === 'SINGLE' && !fieldType.batch ? 2 : 2.5,
  };
};

const isSameEdgeData = (a: WorkflowEdgeData | undefined, b: WorkflowEdgeData): boolean =>
  sameNames(a?.collapsedEdgeIds ?? EMPTY_NAMES, b.collapsedEdgeIds ?? EMPTY_NAMES) &&
  a?.fieldTypeLabel === b.fieldTypeLabel &&
  a.isBatch === b.isBatch &&
  a.isLoopLinkage === b.isLoopLinkage &&
  a.pathType === b.pathType &&
  a.stroke === b.stroke &&
  a.strokeDasharray === b.strokeDasharray &&
  a.strokeWidth === b.strokeWidth;

const isCollapsedInvocation = (node: WorkflowNode | undefined): boolean =>
  node?.type === 'invocation' && !node.data.isOpen;

/** How collapsed nodes reshape a document's edges in the editor; the document itself is never changed. */
export interface CollapsedEdgeLayout {
  /** Nodes drawn collapsed. */
  collapsedNodeIds: ReadonlySet<string>;
  /**
   * Each edge between two collapsed nodes, keyed to every edge between the same pair in document order (legacy's
   * collapsed edge). The first draws for all of them; selecting or deleting any acts on all.
   */
  groupByEdge: ReadonlyMap<string, readonly string[]>;
}

const NO_COLLAPSED_EDGES: CollapsedEdgeLayout = { collapsedNodeIds: new Set(), groupByEdge: new Map() };
const collapsedEdgeLayoutCache = new WeakMap<
  ProjectGraphState['nodes'],
  { edges: ProjectGraphState['edges']; layout: CollapsedEdgeLayout }
>();

/** Cached per document nodes and edges, which change identity only when the graph changes. */
export const getCollapsedEdgeLayout = (document: Pick<ProjectGraphState, 'edges' | 'nodes'>): CollapsedEdgeLayout => {
  const cached = collapsedEdgeLayoutCache.get(document.nodes);

  if (cached?.edges === document.edges) {
    return cached.layout;
  }

  const collapsedNodeIds = new Set(document.nodes.filter(isCollapsedInvocation).map((node) => node.id));
  const groups = new Map<string, string[]>();

  if (collapsedNodeIds.size > 0) {
    for (const edge of document.edges) {
      if (edge.type !== 'loop_linkage' && collapsedNodeIds.has(edge.source) && collapsedNodeIds.has(edge.target)) {
        const key = `${edge.source}\u0000${edge.target}`;
        const group = groups.get(key);

        if (group) {
          group.push(edge.id);
        } else {
          groups.set(key, [edge.id]);
        }
      }
    }
  }

  const groupByEdge = new Map<string, readonly string[]>();

  for (const ids of groups.values()) {
    for (const id of ids) {
      groupByEdge.set(id, ids);
    }
  }

  const layout = collapsedNodeIds.size > 0 ? { collapsedNodeIds, groupByEdge } : NO_COLLAPSED_EDGES;
  collapsedEdgeLayoutCache.set(document.nodes, { edges: document.edges, layout });

  return layout;
};

/** Edge ids widened to every document edge a collapsed edge stands for, so selection and deletion act on all. */
export const withCollapsedEdgeSiblings = (
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  edgeIds: readonly string[]
): string[] => {
  const { groupByEdge } = getCollapsedEdgeLayout(document);

  return groupByEdge.size === 0 ? [...edgeIds] : [...new Set(edgeIds.flatMap((id) => groupByEdge.get(id) ?? [id]))];
};

/**
 * Edge changes with each collapsed group selected as one. A group follows its drawn edge: its state after this batch
 * (its current state, or the change addressed to it) goes to every member. Changes addressed to hidden members are
 * dropped: xyflow reports them only because their state differs, as when a click re-selects the drawn edge and
 * xyflow deselects everything else, so they would otherwise overturn the drawn edge. One lookup per call; the changes
 * pass through untouched when nothing is collapsed.
 */
export const withCollapsedGroupSelection = <Change extends { id?: string; type: string }>(
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  changes: readonly Change[],
  edges: readonly { id: string; selected?: boolean }[]
): (Change | { id: string; selected: boolean; type: 'select' })[] => {
  const { groupByEdge } = getCollapsedEdgeLayout(document);

  if (groupByEdge.size === 0 || !changes.some((change) => change.type === 'select')) {
    return [...changes];
  }

  const drawnChanges = new Map<string, boolean>();
  const touchedGroups = new Set<readonly string[]>();
  const kept = changes.filter((change) => {
    const group = change.type === 'select' && change.id !== undefined ? groupByEdge.get(change.id) : undefined;

    if (!group) {
      return true;
    }

    touchedGroups.add(group);

    if (group[0] === change.id) {
      drawnChanges.set(change.id, (change as unknown as { selected: boolean }).selected);
      return true;
    }

    return false;
  });

  if (touchedGroups.size === 0) {
    return kept;
  }

  const isSelected = new Map(edges.map((edge) => [edge.id, edge.selected ?? false]));
  const memberChanges: { id: string; selected: boolean; type: 'select' }[] = [];

  for (const group of touchedGroups) {
    const drawnId = group[0]!;
    const selected = drawnChanges.get(drawnId) ?? isSelected.get(drawnId) ?? false;

    for (const id of group.slice(1)) {
      if ((isSelected.get(id) ?? false) !== selected) {
        memberChanges.push({ id, selected, type: 'select' });
      }
    }
  }

  return [...kept, ...memberChanges];
};

/**
 * Which ends of an edge may be dragged to reconnect it. An end at a collapsed node's merged port stays put, as legacy's
 * did: that port stands for every field, so a drag there could not say which connection moves, and an edge between two
 * collapsed nodes may stand for several connections. The end at an open node still moves, as one ordinary edge.
 */
const getReconnectableEnds = (
  edge: WorkflowEdge,
  collapsedNodeIds: ReadonlySet<string>
): WorkflowFlowEdge['reconnectable'] => {
  const isSourceCollapsed = collapsedNodeIds.has(edge.source);
  const isTargetCollapsed = collapsedNodeIds.has(edge.target);

  return isSourceCollapsed && isTargetCollapsed
    ? false
    : isSourceCollapsed
      ? 'target'
      : isTargetCollapsed
        ? 'source'
        : undefined;
};

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
  const { collapsedNodeIds, groupByEdge } = getCollapsedEdgeLayout(document);

  return getEdgesWithLoopLinkageAliases(document.nodes, document.edges).map((edge) => {
    const previous = previousById.get(edge.id);
    const collapsedGroup = groupByEdge.get(edge.id);
    const hidden = collapsedGroup !== undefined && collapsedGroup[0] !== edge.id ? true : undefined;
    const typedData = getWorkflowEdgeData(document, edge, edgeType, templates, index);
    // The edge keeps its own type data, so a view drawing every node open (the image export) draws it as itself.
    const data: WorkflowEdgeData = collapsedGroup ? { ...typedData, collapsedEdgeIds: [...collapsedGroup] } : typedData;
    const reconnectable = getReconnectableEnds(edge, collapsedNodeIds);
    // A group is selected as one: its drawn edge and the edges it stands for always agree.
    const selected = collapsedGroup
      ? collapsedGroup.some((id) => previousById.get(id)?.selected)
      : (previous?.selected ?? false);
    const flowType: WorkflowFlowEdgeType = edge.type === 'loop_linkage' ? 'loop_linkage' : edgeType;
    const isConnectedToSelectedNode = selectedNodeIds.has(edge.source) || selectedNodeIds.has(edge.target);
    const animated = isConnectedToSelectedNode && !reduceMotion ? true : undefined;
    const className = isConnectedToSelectedNode ? SELECTED_NODE_EDGE_CLASS : undefined;
    const style = isConnectedToSelectedNode ? SELECTED_NODE_EDGE_STYLE : undefined;

    if (
      previous &&
      previous.type === flowType &&
      previous.source === edge.source &&
      previous.sourceHandle === edge.sourceHandle &&
      previous.target === edge.target &&
      previous.targetHandle === edge.targetHandle &&
      previous.animated === animated &&
      previous.className === className &&
      previous.hidden === hidden &&
      previous.reconnectable === reconnectable &&
      (previous.selected ?? false) === selected &&
      isSameEdgeData(previous.data, data) &&
      previous.style === style
    ) {
      return previous;
    }

    return {
      id: edge.id,
      animated,
      className,
      data,
      hidden,
      reconnectable,
      selected,
      source: edge.source,
      sourceHandle: edge.sourceHandle,
      style,
      target: edge.target,
      targetHandle: edge.targetHandle,
      type: flowType,
    };
  });
};
