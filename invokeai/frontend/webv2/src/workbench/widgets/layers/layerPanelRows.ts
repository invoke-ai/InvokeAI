import type { LayerStackKind } from '@workbench/canvas-engine/api';

import { LAYER_STACKS_TOP_FIRST } from '@workbench/canvas-engine/api';

import type { ProjectedChildRow } from './layerChildRows';
import type { LayerStackRows, LayerStackRowsByKind, LayerTreeRow } from './layerTreeRows';

/** Fixed row height: the virtualizer never measures. */
export const LAYER_ROW_HEIGHT_PX = 40;
export const LAYER_CHILD_ROW_HEIGHT_PX = 32;
export const LAYER_HEADER_HEIGHT_PX = 28;
/** Horizontal offset per nesting level, in CSS pixels; the drag projection uses the same step. */
export const LAYER_TREE_INDENT_PX = 16;
/** Above this many nodes the panel drops thumbnails and drag reordering to stay deterministic. */
export const LAYER_PANEL_DEGRADE_THRESHOLD = 2_000;

export type PanelRow =
  | {
      readonly kind: 'header';
      readonly key: string;
      readonly stack: LayerStackRows;
      readonly collapsed: boolean;
      /** The header's place among the non-empty stacks, for `aria-posinset` / `aria-setsize`. */
      readonly posInSet: number;
      readonly setSize: number;
    }
  | {
      readonly kind: 'node';
      readonly key: string;
      readonly stack: LayerStackKind;
      readonly row: LayerTreeRow;
      /** Projected child rows the node owns; `0` for bare leaves and rowless groups. */
      readonly childCount: number;
      readonly childrenExpanded: boolean;
      /** Position among the parent group's COMBINED children (modifier rows first, then nodes). */
      readonly ariaPosInSet: number;
      readonly ariaSetSize: number;
    }
  | {
      readonly kind: 'child';
      readonly key: string;
      readonly stack: LayerStackKind;
      readonly child: ProjectedChildRow;
      /** The owner's combined child count (rows + node children) for a group owner. */
      readonly ariaSetSize: number;
    };

/** How a flattening projects child rows beneath the layers that own them. */
export interface PanelChildRowsSource {
  rowsFor(row: LayerTreeRow): readonly ProjectedChildRow[];
  /** Layers whose child rows are hidden; every other owner shows them. */
  collapsedLayerIds: ReadonlySet<string>;
}

export const headerKey = (stack: LayerStackKind): string => `header:${stack}`;

export const isHeaderKey = (key: string): boolean => key.startsWith('header:');

export const stackOfHeaderKey = (key: string): LayerStackKind => key.slice('header:'.length) as LayerStackKind;

/**
 * Flatten stack headers and visible rows for virtualization; forceOpen keeps pending targets reachable inside
 * collapsed stacks.
 */
export const flattenPanelRows = (
  stacks: LayerStackRowsByKind,
  collapsedStacks: readonly LayerStackKind[],
  forceOpen: (stack: LayerStackKind) => boolean,
  childRows?: PanelChildRowsSource
): PanelRow[] => {
  const rows: PanelRow[] = [];
  const present = LAYER_STACKS_TOP_FIRST.filter((kind) => stacks[kind].nodeIds.length > 0);
  for (const kind of present) {
    const stack = stacks[kind];
    const collapsed = collapsedStacks.includes(kind) && !forceOpen(kind);
    rows.push({
      collapsed,
      key: headerKey(kind),
      kind: 'header',
      posInSet: present.indexOf(kind) + 1,
      setSize: present.length,
      stack,
    });
    if (!collapsed) {
      const rowOffsetByParent = new Map<string, number>();
      for (const row of stack.rows) {
        const isGroup = row.vm.kind === 'group';
        // A group's rows fold WITH its subtree, keeping aria-expanded truthful.
        const children = isGroup && !row.expanded ? [] : (childRows?.rowsFor(row) ?? []);
        const childrenExpanded = children.length > 0 && (isGroup || !childRows!.collapsedLayerIds.has(row.id));
        const offset = rowOffsetByParent.get(row.vm.parentId ?? '') ?? 0;
        if (isGroup && childrenExpanded) {
          rowOffsetByParent.set(row.id, children.length);
        }
        rows.push({
          ariaPosInSet: row.posInSet + offset,
          ariaSetSize: row.setSize + offset,
          childCount: children.length,
          childrenExpanded,
          key: row.id,
          kind: 'node',
          row,
          stack: kind,
        });
        if (childrenExpanded) {
          const combined = isGroup ? children.length + row.vm.childCount : children.length;
          for (const child of children) {
            rows.push({ ariaSetSize: combined, child, key: child.key, kind: 'child', stack: kind });
          }
        }
      }
    }
  }
  return rows;
};

export const panelRowHeight = (row: PanelRow): number =>
  row.kind === 'header'
    ? LAYER_HEADER_HEIGHT_PX
    : row.kind === 'child'
      ? LAYER_CHILD_ROW_HEIGHT_PX
      : LAYER_ROW_HEIGHT_PX;

export type TreeNavigationKey = 'ArrowDown' | 'ArrowUp' | 'Home' | 'End' | 'ArrowLeft' | 'ArrowRight';

export type TreeNavigation =
  | { readonly focus: string }
  | { readonly expand: string; readonly expanded: boolean }
  | { readonly expandChildren: string; readonly expanded: boolean }
  | { readonly collapseStack: LayerStackKind; readonly collapsed: boolean };

const positionsCache = new WeakMap<readonly PanelRow[], Map<string, number>>();

const positionsOf = (rows: readonly PanelRow[]): Map<string, number> => {
  let cached = positionsCache.get(rows);
  if (!cached) {
    cached = new Map(rows.map((row, index) => [row.key, index]));
    positionsCache.set(rows, cached);
  }
  return cached;
};

/**
 * Apply tree keyboard navigation across headers/rows: vertical/Home/End traversal, Right open/enter, Left
 * close/parent.
 */
export const navigateTree = (
  rows: readonly PanelRow[],
  currentKey: string,
  key: TreeNavigationKey
): TreeNavigation | null => {
  const position = positionsOf(rows).get(currentKey) ?? -1;
  if (position < 0) {
    return rows[0] ? { focus: rows[0].key } : null;
  }
  const current = rows[position]!;
  switch (key) {
    case 'ArrowDown':
      return rows[position + 1] ? { focus: rows[position + 1]!.key } : null;
    case 'ArrowUp':
      return rows[position - 1] ? { focus: rows[position - 1]!.key } : null;
    case 'Home':
      return { focus: rows[0]!.key };
    case 'End':
      return { focus: rows[rows.length - 1]!.key };
    case 'ArrowRight': {
      if (current.kind === 'header') {
        if (current.collapsed) {
          return { collapsed: false, collapseStack: current.stack.stack };
        }
        const first = rows[position + 1];
        return first && first.kind === 'node' ? { focus: first.key } : null;
      }
      if (current.kind === 'child') {
        return null;
      }
      const { row } = current;
      if (row.vm.kind === 'group' && row.vm.childCount > 0 && !row.expanded) {
        return { expand: row.id, expanded: true };
      }
      if (current.childCount > 0 && !current.childrenExpanded) {
        return { expandChildren: row.id, expanded: true };
      }
      const next = rows[position + 1];
      if (next && next.kind === 'child' && next.child.layerId === row.id) {
        return { focus: next.key };
      }
      if (next && next.kind === 'node' && next.row.vm.parentId === row.id) {
        return { focus: next.key };
      }
      return null;
    }
    case 'ArrowLeft': {
      if (current.kind === 'header') {
        return current.collapsed ? null : { collapsed: true, collapseStack: current.stack.stack };
      }
      if (current.kind === 'child') {
        return { focus: current.child.layerId };
      }
      const { row } = current;
      if (row.vm.kind !== 'group' && current.childrenExpanded) {
        return { expandChildren: row.id, expanded: false };
      }
      if (row.vm.kind === 'group' && row.expanded) {
        return { expand: row.id, expanded: false };
      }
      return { focus: row.vm.parentId ?? headerKey(row.vm.stack) };
    }
  }
};

export const isTreeNavigationKey = (key: string): key is TreeNavigationKey =>
  key === 'ArrowDown' ||
  key === 'ArrowUp' ||
  key === 'Home' ||
  key === 'End' ||
  key === 'ArrowLeft' ||
  key === 'ArrowRight';
