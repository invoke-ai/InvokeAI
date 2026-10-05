import type { LayerStackKind } from '@workbench/canvas-engine/api';
import type { LayerSelectionModifiers } from '@workbench/layerPanelState';
import type { KeyboardEvent } from 'react';

import type { ProjectedChildRow } from './layerChildRows';

/** A point or box the surface host anchors a menu or popover to, in viewport pixels. */
export interface LayerSurfaceAnchor {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/**
 * Share one stable command handle per tree; rows rerender for their own model/selection while the tree owns
 * subscriptions and commits.
 */
export interface LayerRowCommands {
  select(id: string, modifiers: LayerSelectionModifiers): void;
  focus(id: string): void;
  toggleExpanded(id: string): void;
  setEnabled(id: string, isEnabled: boolean): void;
  setHidden(id: string, isHidden: boolean): void;
  setLocked(id: string, isLocked: boolean): void;
  startRename(id: string): void;
  rename(id: string, name: string): void;
  endRename(): void;
  openMenu(id: string, anchor: LayerSurfaceAnchor): void;
  openStackMenu(stack: LayerStackKind, anchor: LayerSurfaceAnchor): void;
  toggleCollapse(stack: LayerStackKind): void;
  /** Shows or hides a layer's projected child rows. */
  toggleChildren(id: string): void;
  /**
   * Sub-selects a child row: its owner becomes the selected layer and its
   * editor opens in Properties, unless `reveal: false` (a context menu selects
   * without yanking the pane open).
   */
  selectChild(child: ProjectedChildRow, options?: { reveal?: boolean }): void;
  setChildEnabled(child: ProjectedChildRow, isEnabled: boolean): void;
  removeChild(child: ProjectedChildRow): void;
  /** Renameable kinds (adjustment entries): sets the custom name, `null` restoring the kind's own. */
  renameChild(child: ProjectedChildRow, name: string | null): void;
  /** Ordered kinds only (adjustment entries): swap with the neighbour in `direction`. */
  moveChild(child: ProjectedChildRow, direction: -1 | 1): void;
  /** Reference images only: append the item to another regional layer as one atomic edit. */
  moveChildToLayer(child: ProjectedChildRow, layerId: string): void;
  /** Ordered kinds only: insert a copy directly after the entry. */
  duplicateChild(child: ProjectedChildRow): void;
  openChildMenu(child: ProjectedChildRow, anchor: LayerSurfaceAnchor): void;
  /** `id` is a node id, a child-row key, or a stack header key; the tree tells them apart. */
  keyDown(id: string, event: KeyboardEvent<HTMLElement>): void;
}

export const anchorFromRect = (rect: DOMRect): LayerSurfaceAnchor => ({
  height: rect.height,
  width: rect.width,
  x: rect.left,
  y: rect.top,
});

export const anchorFromPoint = (x: number, y: number): LayerSurfaceAnchor => ({ height: 1, width: 1, x, y });
