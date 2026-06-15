import { createExternalStore } from '@workbench/externalStore';
import { createWorkflowId } from '@workbench/workflows/document';
import type { ProjectGraphState, WorkflowEdge, WorkflowNode, XYPosition } from '@workbench/workflows/types';

/**
 * Session-lived clipboard for workflow nodes. Copies hold deep clones of the
 * selected nodes plus the edges that connect them to each other; pasting
 * materializes fresh ids (remapping those internal edges) so a payload can be
 * pasted repeatedly. Deliberately not the system clipboard: node fragments are
 * not a valid workflow JSON, so a partial graph on the OS clipboard would only
 * masquerade as an importable document.
 */

interface WorkflowClipboardSnapshot {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

const clipboardStore = createExternalStore<WorkflowClipboardSnapshot>({ edges: [], nodes: [] });

const PASTE_OFFSET = 32;

export const copyNodesToClipboard = (document: ProjectGraphState, nodeIds: string[]): number => {
  const copiedIds = new Set(nodeIds);
  const nodes = document.nodes.filter((node) => copiedIds.has(node.id));

  if (nodes.length === 0) {
    return 0;
  }

  const edges = document.edges.filter((edge) => copiedIds.has(edge.source) && copiedIds.has(edge.target));

  clipboardStore.setSnapshot(structuredClone({ edges, nodes }));

  return nodes.length;
};

export const hasClipboardNodes = (): boolean => clipboardStore.getSnapshot().nodes.length > 0;

export const useHasClipboardNodes = (): boolean => clipboardStore.useSnapshot().nodes.length > 0;

const materializeElements = (
  payload: WorkflowClipboardSnapshot,
  at?: XYPosition
): { nodes: WorkflowNode[]; edges: WorkflowEdge[] } => {
  const { edges, nodes } = structuredClone(payload);

  if (nodes.length === 0) {
    return { edges: [], nodes: [] };
  }

  const idMap = new Map<string, string>();

  for (const node of nodes) {
    idMap.set(node.id, createWorkflowId(node.type === 'invocation' ? node.data.type : node.type));
  }

  let offset: XYPosition = { x: PASTE_OFFSET, y: PASTE_OFFSET };

  if (at) {
    const minX = Math.min(...nodes.map((node) => node.position.x));
    const minY = Math.min(...nodes.map((node) => node.position.y));

    offset = { x: at.x - minX, y: at.y - minY };
  }

  return {
    edges: edges.map((edge) => ({
      ...edge,
      id: createWorkflowId('edge'),
      source: idMap.get(edge.source) ?? edge.source,
      target: idMap.get(edge.target) ?? edge.target,
    })),
    nodes: nodes.map((node) => ({
      ...node,
      id: idMap.get(node.id) ?? node.id,
      position: { x: node.position.x + offset.x, y: node.position.y + offset.y },
    })),
  };
};

/**
 * Materializes the clipboard into paste-ready elements with fresh ids.
 * Positions shift by a fixed offset from the copied originals (or anchor the
 * group's top-left on `at` when given) so pastes do not stack on their source.
 */
export const buildPasteElements = (at?: XYPosition): { nodes: WorkflowNode[]; edges: WorkflowEdge[] } =>
  materializeElements(clipboardStore.getSnapshot(), at);

/** Clone-in-place for the given nodes (and their internal edges) without touching the clipboard. */
export const buildDuplicateElements = (
  document: ProjectGraphState,
  nodeIds: string[]
): { nodes: WorkflowNode[]; edges: WorkflowEdge[] } => {
  const copiedIds = new Set(nodeIds);

  return materializeElements({
    edges: document.edges.filter((edge) => copiedIds.has(edge.source) && copiedIds.has(edge.target)),
    nodes: document.nodes.filter((node) => copiedIds.has(node.id)),
  });
};
