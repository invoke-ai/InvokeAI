import type { ProjectGraphState, WorkflowEdge, WorkflowNode, XYPosition } from '@features/workflow/contracts';

import { isConnectorNode } from '@features/workflow/contracts';
import {
  createWorkflowGraphIndex,
  createWorkflowId,
  getConnectorInputEdgeIndexed,
  getConnectorOutputEdgesIndexed,
  LOOP_LINKAGE_FIELD,
  type WorkflowGraphIndex,
} from '@features/workflow/utility';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Keep deep-cloned node fragments in a session clipboard; paste remaps internal edges to fresh IDs. Fragments are
 * not valid system-clipboard workflow documents.
 */

interface WorkflowClipboardSnapshot {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  /** Edges from nodes outside the copy into it, kept by their original source id for paste with connections. */
  incomingEdges: WorkflowEdge[];
}

interface WorkflowGraphElements {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

/** Where a pasted group lands: its top-left node position, or the middle of its node positions, at `point`. */
export interface WorkflowPastePlacement {
  align: 'center' | 'top-left';
  point: XYPosition;
}

const EMPTY_SNAPSHOT: WorkflowClipboardSnapshot = { edges: [], incomingEdges: [], nodes: [] };

const clipboardStore = createExternalStore<WorkflowClipboardSnapshot>(EMPTY_SNAPSHOT);

const PASTE_OFFSET = 32;

export const clearWorkflowClipboard = (): void => {
  clipboardStore.setSnapshot(EMPTY_SNAPSHOT);
};

registerAccountOwnedResource({
  clear: clearWorkflowClipboard,
  name: 'workflow-clipboard',
});

/**
 * Unselected connectors that carry a copied node's output on to another copied node. Copying them keeps A → C → B
 * intact when only A and B are selected.
 */
const getBridgingConnectorIds = (
  nodes: WorkflowNode[],
  index: WorkflowGraphIndex,
  copiedIds: ReadonlySet<string>
): Set<string> => {
  const upstreamCache = new Map<string, boolean>();
  const downstreamCache = new Map<string, boolean>();

  const isConnectorId = (id: string): boolean => {
    const node = index.nodesById.get(id);
    return node !== undefined && isConnectorNode(node);
  };

  const startsAtCopiedNode = (connectorId: string, visiting: Set<string>): boolean => {
    const cached = upstreamCache.get(connectorId);

    if (cached !== undefined) {
      return cached;
    }

    const source = getConnectorInputEdgeIndexed(connectorId, index)?.source;
    let result = false;

    if (source !== undefined && !visiting.has(source)) {
      visiting.add(source);
      result = copiedIds.has(source) || (isConnectorId(source) && startsAtCopiedNode(source, visiting));
    }

    upstreamCache.set(connectorId, result);
    return result;
  };

  const reachesCopiedNode = (connectorId: string, visiting: Set<string>): boolean => {
    const cached = downstreamCache.get(connectorId);

    if (cached !== undefined) {
      return cached;
    }

    let result = false;

    for (const edge of getConnectorOutputEdgesIndexed(connectorId, index)) {
      if (visiting.has(edge.target)) {
        continue;
      }

      visiting.add(edge.target);

      if (copiedIds.has(edge.target) || (isConnectorId(edge.target) && reachesCopiedNode(edge.target, visiting))) {
        result = true;
        break;
      }
    }

    downstreamCache.set(connectorId, result);
    return result;
  };

  const bridgingIds = new Set<string>();

  for (const node of nodes) {
    if (
      isConnectorNode(node) &&
      !copiedIds.has(node.id) &&
      startsAtCopiedNode(node.id, new Set([node.id])) &&
      reachesCopiedNode(node.id, new Set([node.id]))
    ) {
      bridgingIds.add(node.id);
    }
  }

  return bridgingIds;
};

const isLoopLinkageEdge = (edge: WorkflowEdge): boolean =>
  edge.type === 'loop_linkage' || edge.targetHandle === LOOP_LINKAGE_FIELD;

/** Copies the nodes, their internal edges, bridging connectors, and incoming edges; returns the selected count. */
export const copyNodesToClipboard = (document: ProjectGraphState, nodeIds: string[]): number => {
  const selectedIds = new Set(nodeIds);
  const selectedCount = document.nodes.filter((node) => selectedIds.has(node.id)).length;

  if (selectedCount === 0) {
    return 0;
  }

  const index = createWorkflowGraphIndex(document.nodes, document.edges);
  const copiedIds = new Set([...selectedIds, ...getBridgingConnectorIds(document.nodes, index, selectedIds)]);
  const nodes = document.nodes.filter((node) => copiedIds.has(node.id));
  const edges: WorkflowEdge[] = [];
  const incomingEdges: WorkflowEdge[] = [];

  for (const edge of document.edges) {
    if (!copiedIds.has(edge.target)) {
      continue;
    }

    if (copiedIds.has(edge.source)) {
      edges.push(edge);
    } else if (!isLoopLinkageEdge(edge)) {
      // A For links to exactly one ForReturn, so a pasted ForReturn never takes the original's linkage.
      incomingEdges.push(edge);
    }
  }

  clipboardStore.setSnapshot(structuredClone({ edges, incomingEdges, nodes }));

  return selectedCount;
};

export const hasClipboardNodes = (): boolean => clipboardStore.getSnapshot().nodes.length > 0;

export const useHasClipboardNodes = (): boolean => clipboardStore.useSelector((snapshot) => snapshot.nodes.length > 0);

const getPlacementOffset = (nodes: WorkflowNode[], placement: WorkflowPastePlacement | undefined): XYPosition => {
  if (!placement) {
    return { x: PASTE_OFFSET, y: PASTE_OFFSET };
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const { position } of nodes) {
    minX = Math.min(minX, position.x);
    minY = Math.min(minY, position.y);
    maxX = Math.max(maxX, position.x);
    maxY = Math.max(maxY, position.y);
  }

  const anchor = placement.align === 'center' ? { x: (minX + maxX) / 2, y: (minY + maxY) / 2 } : { x: minX, y: minY };

  return { x: placement.point.x - anchor.x, y: placement.point.y - anchor.y };
};

const materializeElements = (
  payload: WorkflowClipboardSnapshot,
  { placement, withConnections = false }: { placement?: WorkflowPastePlacement; withConnections?: boolean } = {}
): WorkflowGraphElements => {
  const { edges, incomingEdges, nodes } = structuredClone(payload);

  if (nodes.length === 0) {
    return { edges: [], nodes: [] };
  }

  const idMap = new Map<string, string>();

  for (const node of nodes) {
    idMap.set(node.id, createWorkflowId(node.type === 'invocation' ? node.data.type : node.type));
  }

  const offset = getPlacementOffset(nodes, placement);
  const internalEdges = edges.map((edge) => ({
    ...edge,
    id: createWorkflowId('edge'),
    source: idMap.get(edge.source) ?? edge.source,
    target: idMap.get(edge.target) ?? edge.target,
  }));
  // Incoming edges keep their outside source; the reducer drops any whose source is not in the document.
  const connectedEdges = withConnections
    ? incomingEdges.map((edge) => ({
        ...edge,
        id: createWorkflowId('edge'),
        target: idMap.get(edge.target) ?? edge.target,
      }))
    : [];

  return {
    edges: [...internalEdges, ...connectedEdges],
    nodes: nodes.map((node) => ({
      ...node,
      id: idMap.get(node.id) ?? node.id,
      position: { x: node.position.x + offset.x, y: node.position.y + offset.y },
    })),
  };
};

/**
 * Assign fresh IDs and place the group (a fixed offset from the original without a placement). With connections,
 * edges that fed the copied nodes from outside feed the pasted copies from the same sources.
 */
export const buildPasteElements = (
  options: { placement?: WorkflowPastePlacement; withConnections?: boolean } = {}
): WorkflowGraphElements => materializeElements(clipboardStore.getSnapshot(), options);

/** Clone-in-place for the given nodes (and their internal edges) without touching the clipboard. */
export const buildDuplicateElements = (document: ProjectGraphState, nodeIds: string[]): WorkflowGraphElements => {
  const copiedIds = new Set(nodeIds);

  return materializeElements({
    edges: document.edges.filter((edge) => copiedIds.has(edge.source) && copiedIds.has(edge.target)),
    incomingEdges: [],
    nodes: document.nodes.filter((node) => copiedIds.has(node.id)),
  });
};
