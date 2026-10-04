import type { CanvasStackForests, LayerStackKind, SemanticNode } from '@workbench/canvas-engine/api';

import { compileDocumentNodes, LAYER_STACKS_TOP_FIRST } from '@workbench/canvas-engine/api';

/** One Layers-panel row: the seam's node view model plus what the panel adds to it. */
export interface LayerTreeRow {
  readonly id: string;
  readonly vm: SemanticNode;
  /** Groups only: whether the row shows its children. */
  readonly expanded: boolean;
  /** 1-based position among the node's siblings, and how many siblings there are. */
  readonly posInSet: number;
  readonly setSize: number;
}

export interface LayerStackRows {
  readonly stack: LayerStackKind;
  /** Rows the panel renders, top first; children of collapsed groups and filtered-out nodes are absent. */
  readonly rows: readonly LayerTreeRow[];
  /** Every node id in the stack, top first, whether rendered or not. */
  readonly nodeIds: readonly string[];
  readonly leafCount: number;
  readonly groupCount: number;
}

export type LayerStackRowsByKind = Record<LayerStackKind, LayerStackRows>;

const rowsByNode = new WeakMap<SemanticNode, LayerTreeRow>();

const rowFor = (vm: SemanticNode, expanded: boolean, posInSet: number, setSize: number): LayerTreeRow => {
  const cached = rowsByNode.get(vm);
  if (cached && cached.expanded === expanded && cached.posInSet === posInSet && cached.setSize === setSize) {
    return cached;
  }
  const row: LayerTreeRow = { expanded, id: vm.id, posInSet, setSize, vm };
  rowsByNode.set(vm, row);
  return row;
};

const normalizeFilter = (filter: string | undefined): string => (filter ?? '').trim().toLocaleLowerCase();

/**
 * The nodes a filter keeps: every node whose name matches, plus the groups above a match so the
 * match stays reachable. A kept group shows expanded when something kept sits beneath it; a
 * matching group with no kept descendant shows collapsed, since its children are not matches.
 */
const filteredIds = (
  nodes: readonly SemanticNode[],
  filter: string
): { kept: ReadonlySet<string>; open: ReadonlySet<string> } => {
  const kept = new Set<string>();
  const open = new Set<string>();
  for (const node of nodes) {
    if (node.node.name.toLocaleLowerCase().includes(filter)) {
      kept.add(node.id);
      for (const ancestor of node.parentIds) {
        kept.add(ancestor);
        open.add(ancestor);
      }
    }
  }
  return { kept, open };
};

/**
 * Project each node once and preserve row identity while semantic state/placement/expansion remain unchanged so
 * unaffected rows skip renders.
 */
export const buildLayerStackRows = (
  stacks: CanvasStackForests,
  expandedGroupIds: ReadonlySet<string>,
  filter?: string
): LayerStackRowsByKind => {
  const nodes = compileDocumentNodes({ stacks });
  const query = normalizeFilter(filter);
  const filtered = query ? filteredIds(nodes, query) : null;
  const kept = filtered?.kept ?? null;
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const setSizeOf = (node: SemanticNode): number =>
    node.parentId === null ? stacks[node.stack].length : (byId.get(node.parentId)?.childCount ?? 0);
  const result = Object.fromEntries(
    LAYER_STACKS_TOP_FIRST.map((stack) => [
      stack,
      { groupCount: 0, leafCount: 0, nodeIds: [] as string[], rows: [] as LayerTreeRow[], stack },
    ])
  ) as Record<
    LayerStackKind,
    { groupCount: number; leafCount: number; nodeIds: string[]; rows: LayerTreeRow[]; stack: LayerStackKind }
  >;
  const collapsed = new Set<string>();
  for (const node of nodes) {
    const target = result[node.stack];
    target.nodeIds.push(node.id);
    if (node.kind === 'group') {
      target.groupCount += 1;
    } else {
      target.leafCount += 1;
    }
    if (kept && !kept.has(node.id)) {
      continue;
    }
    if (node.parentIds.some((ancestor) => collapsed.has(ancestor))) {
      if (node.kind === 'group' && !expandedGroupIds.has(node.id)) {
        collapsed.add(node.id);
      }
      continue;
    }
    const expanded = node.kind === 'group' && (filtered ? filtered.open.has(node.id) : expandedGroupIds.has(node.id));
    if (node.kind === 'group' && !expanded) {
      collapsed.add(node.id);
    }
    target.rows.push(rowFor(node, expanded, node.siblingIndex + 1, setSizeOf(node)));
  }
  return result;
};
