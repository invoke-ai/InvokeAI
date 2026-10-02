import type { CanvasDocumentContractV3, CanvasDocumentIndex, LayerStackKind } from '@workbench/canvas-engine/api';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { getDocumentIndex } from '@workbench/canvas-engine/api';

/**
 * Per-project transient selection stays outside persistence/history; primaryId detects external primary changes
 * and clears stale secondaries.
 */

export interface LayerPanelState {
  readonly projectId: string;
  readonly primaryId: string | null;
  readonly anchorId: string | null;
  readonly selectedIds: readonly string[];
  readonly collapsedStacks: readonly LayerStackKind[];
  /** Groups whose children the panel shows; every other group is collapsed. */
  readonly expandedGroupIds: readonly string[];
  /** Layers whose projected child rows are hidden; every other owner shows them. */
  readonly collapsedChildLayerIds: readonly string[];
  /** The row that holds keyboard focus (roving tabindex); falls back to the primary. */
  readonly focusId: string | null;
  /** A name filter; empty shows everything. */
  readonly filter: string;
}

/** Panel preferences that survive a primary change and a project switch. */
type LayerPanelCarry = Pick<
  LayerPanelState,
  'collapsedStacks' | 'expandedGroupIds' | 'collapsedChildLayerIds' | 'filter'
>;

const DEFAULT_CARRY: LayerPanelCarry = {
  collapsedChildLayerIds: [],
  collapsedStacks: [],
  expandedGroupIds: [],
  filter: '',
};

export interface LayerSelectionModifiers {
  additive: boolean;
  range: boolean;
}

/** Tree key of a projected child row; the panel's focus state holds these beside node ids. */
export const layerChildRowKey = (layerId: string, itemId: string): string => `child:${layerId}:${itemId}`;

export const parseChildRowKey = (key: string): { layerId: string; itemId: string } | null => {
  if (!key.startsWith('child:')) {
    return null;
  }
  const rest = key.slice('child:'.length);
  const separator = rest.indexOf(':');
  return separator > 0 ? { itemId: rest.slice(separator + 1), layerId: rest.slice(0, separator) } : null;
};

export interface LayerPanelSelectionUpdate {
  projectId: string;
  primaryId: string | null;
  selectedIds: readonly string[];
  anchorId?: string | null;
}

interface LayerPanelStore {
  readonly byProject: Readonly<Record<string, LayerPanelState>>;
}

const store = createExternalStore<LayerPanelStore>({ byProject: {} });

export const createLayerPanelState = (
  projectId: string,
  primaryId: string | null,
  carry: Partial<LayerPanelCarry> = {}
): LayerPanelState => ({
  ...DEFAULT_CARRY,
  ...carry,
  anchorId: primaryId,
  focusId: primaryId,
  primaryId,
  projectId,
  selectedIds: primaryId ? [primaryId] : [],
});

const carryOf = (state: LayerPanelState): LayerPanelCarry => ({
  collapsedChildLayerIds: state.collapsedChildLayerIds,
  collapsedStacks: state.collapsedStacks,
  expandedGroupIds: state.expandedGroupIds,
  filter: state.filter,
});

const sameIds = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((id, index) => id === right[index]);

export const isSameLayerPanelState = (left: LayerPanelState, right: LayerPanelState): boolean =>
  left.projectId === right.projectId &&
  left.primaryId === right.primaryId &&
  left.anchorId === right.anchorId &&
  left.focusId === right.focusId &&
  left.filter === right.filter &&
  sameIds(left.selectedIds, right.selectedIds) &&
  sameIds(left.collapsedStacks, right.collapsedStacks) &&
  sameIds(left.expandedGroupIds, right.expandedGroupIds) &&
  sameIds(left.collapsedChildLayerIds, right.collapsedChildLayerIds);

const stateFor = (snapshot: LayerPanelStore, projectId: string, primaryId: string | null): LayerPanelState => {
  const stored = snapshot.byProject[projectId];
  if (!stored) {
    return createLayerPanelState(projectId, primaryId);
  }
  return stored.primaryId === primaryId ? stored : createLayerPanelState(projectId, primaryId, carryOf(stored));
};

export const readLayerPanelState = (projectId: string, primaryId: string | null): LayerPanelState =>
  stateFor(store.getSnapshot(), projectId, primaryId);

export const useLayerPanelState = (projectId: string, primaryId: string | null): LayerPanelState =>
  store.useSelector((snapshot) => stateFor(snapshot, projectId, primaryId), isSameLayerPanelState);

const write = (state: LayerPanelState): void => {
  const { byProject } = store.getSnapshot();
  const current = byProject[state.projectId];
  if (current && isSameLayerPanelState(current, state)) {
    return;
  }
  store.setSnapshot({ byProject: { ...byProject, [state.projectId]: state } });
};

/** Records a panel-originated selection; publish before dispatching a new primary so reconciliation keeps it. */
export const publishLayerPanelSelection = (selection: LayerPanelSelectionUpdate): void => {
  const stored = store.getSnapshot().byProject[selection.projectId];
  write({
    ...(stored ? carryOf(stored) : DEFAULT_CARRY),
    anchorId: selection.anchorId ?? selection.primaryId,
    focusId: selection.primaryId,
    primaryId: selection.primaryId,
    projectId: selection.projectId,
    selectedIds: [...selection.selectedIds],
  });
};

export const toggleLayerStackCollapsed = (projectId: string, primaryId: string | null, stack: LayerStackKind): void => {
  const current = readLayerPanelState(projectId, primaryId);
  const collapsedStacks = current.collapsedStacks.includes(stack)
    ? current.collapsedStacks.filter((candidate) => candidate !== stack)
    : [...current.collapsedStacks, stack];
  write({ ...current, collapsedStacks });
};

/** Shows or hides a group's children; `expanded` forces a state instead of toggling. */
export const setLayerGroupExpanded = (
  projectId: string,
  primaryId: string | null,
  groupIds: readonly string[],
  expanded?: boolean
): void => {
  const current = readLayerPanelState(projectId, primaryId);
  const next = new Set(current.expandedGroupIds);
  for (const groupId of groupIds) {
    if (expanded ?? !next.has(groupId)) {
      next.add(groupId);
    } else {
      next.delete(groupId);
    }
  }
  write({ ...current, expandedGroupIds: [...next] });
};

/** Shows or hides a layer's projected child rows; `collapsed` forces a state instead of toggling. */
export const setLayerChildrenCollapsed = (
  projectId: string,
  primaryId: string | null,
  layerId: string,
  collapsed?: boolean
): void => {
  const current = readLayerPanelState(projectId, primaryId);
  const next = new Set(current.collapsedChildLayerIds);
  if (collapsed ?? !next.has(layerId)) {
    next.add(layerId);
  } else {
    next.delete(layerId);
  }
  write({ ...current, collapsedChildLayerIds: [...next] });
};

export const setLayerPanelFocus = (projectId: string, primaryId: string | null, focusId: string | null): void => {
  const current = readLayerPanelState(projectId, primaryId);
  if (current.focusId !== focusId) {
    write({ ...current, focusId });
  }
};

export const setLayerPanelFilter = (projectId: string, primaryId: string | null, filter: string): void => {
  const current = readLayerPanelState(projectId, primaryId);
  if (current.filter !== filter) {
    write({ ...current, filter });
  }
};

/** Reconcile external changes and removals; reveal an externally selected primary by expanding its ancestors. */
export const reconcileLayerPanelState = (
  state: LayerPanelState,
  projectId: string,
  index: CanvasDocumentIndex,
  primaryId: string | null
): LayerPanelState => {
  const primary = primaryId ? index.byId.get(primaryId) : undefined;
  const validPrimaryId = primary ? primary.node.id : null;
  if (state.projectId !== projectId || state.primaryId !== validPrimaryId) {
    if (state.projectId !== projectId) {
      return createLayerPanelState(projectId, validPrimaryId, { expandedGroupIds: primary ? [...primary.path] : [] });
    }
    // The new primary is revealed: its stack opens and every group on its path expands.
    const carry = carryOf(state);
    return createLayerPanelState(projectId, validPrimaryId, {
      ...carry,
      collapsedChildLayerIds: carry.collapsedChildLayerIds.filter((id) => index.byId.has(id)),
      collapsedStacks: primary
        ? carry.collapsedStacks.filter((stack) => stack !== primary.stack)
        : carry.collapsedStacks,
      expandedGroupIds: [
        ...new Set([...carry.expandedGroupIds.filter((id) => index.byId.has(id)), ...(primary?.path ?? [])]),
      ],
    });
  }
  const existing = index.byId;
  const expandedGroupIds = state.expandedGroupIds.filter((id) => existing.has(id));
  const collapsedChildLayerIds = state.collapsedChildLayerIds.filter((id) => existing.has(id));
  const selectedIds = [...new Set(state.selectedIds)].filter((id) => existing.has(id));
  if (validPrimaryId && !selectedIds.includes(validPrimaryId)) {
    selectedIds.push(validPrimaryId);
  }
  const anchorId = state.anchorId && existing.has(state.anchorId) ? state.anchorId : validPrimaryId;
  // A child-row focus survives while its owner exists; the panel falls back itself if the item is gone.
  const focusOwnerId = state.focusId ? (parseChildRowKey(state.focusId)?.layerId ?? state.focusId) : null;
  const focusId = state.focusId && focusOwnerId && existing.has(focusOwnerId) ? state.focusId : validPrimaryId;
  if (
    anchorId === state.anchorId &&
    focusId === state.focusId &&
    sameIds(selectedIds, state.selectedIds) &&
    sameIds(expandedGroupIds, state.expandedGroupIds) &&
    sameIds(collapsedChildLayerIds, state.collapsedChildLayerIds)
  ) {
    return state;
  }
  return { ...state, anchorId, collapsedChildLayerIds, expandedGroupIds, focusId, selectedIds };
};

/**
 * Applies plain, Ctrl/Cmd-toggle, and Shift-range row selection semantics over the rows the panel
 * renders; an additive range keeps selected rows hidden inside collapsed stacks.
 */
export const selectLayerInPanel = (
  state: LayerPanelState,
  layerId: string,
  orderedIds: readonly string[],
  modifiers: LayerSelectionModifiers
): LayerPanelState => {
  if (!orderedIds.includes(layerId)) {
    return state;
  }
  if (modifiers.range) {
    const anchorId = state.anchorId && orderedIds.includes(state.anchorId) ? state.anchorId : layerId;
    const start = orderedIds.indexOf(anchorId);
    const end = orderedIds.indexOf(layerId);
    const rangeIds = orderedIds.slice(Math.min(start, end), Math.max(start, end) + 1);
    const selected = new Set(modifiers.additive ? [...state.selectedIds, ...rangeIds] : rangeIds);
    const hidden = modifiers.additive ? state.selectedIds.filter((id) => !orderedIds.includes(id)) : [];
    return {
      ...state,
      anchorId,
      primaryId: layerId,
      selectedIds: [...orderedIds.filter((id) => selected.has(id)), ...hidden],
    };
  }
  if (modifiers.additive) {
    const selected = new Set(state.selectedIds);
    const wasSelected = selected.has(layerId);
    if (wasSelected) {
      selected.delete(layerId);
    } else {
      selected.add(layerId);
    }
    const selectedIds = orderedIds.filter((id) => selected.has(id));
    const primaryId = wasSelected
      ? state.primaryId === layerId || !state.primaryId || !selected.has(state.primaryId)
        ? (selectedIds[0] ?? null)
        : state.primaryId
      : layerId;
    return { ...state, anchorId: layerId, primaryId, selectedIds };
  }
  return { ...state, anchorId: layerId, primaryId: layerId, selectedIds: [layerId] };
};

export interface LayerPanelProjectView {
  readonly id: string;
  readonly canvas: { readonly document: Pick<CanvasDocumentContractV3, 'stacks' | 'selectedLayerId'> };
}

const reconciledDocuments = new Map<string, LayerPanelProjectView['canvas']['document']>();

/** Reconciles every stored state with its project after a store transition and forgets closed projects. */
export const reconcileLayerPanelStates = (projects: readonly LayerPanelProjectView[]): void => {
  const { byProject } = store.getSnapshot();
  const next: Record<string, LayerPanelState> = {};
  let changed = false;
  for (const project of projects) {
    const stored = byProject[project.id] ?? createLayerPanelState(project.id, null);
    const { document } = project.canvas;
    if (reconciledDocuments.get(project.id) === document) {
      next[project.id] = stored;
      continue;
    }
    reconciledDocuments.set(project.id, document);
    const reconciled = reconcileLayerPanelState(
      stored,
      project.id,
      getDocumentIndex(document),
      document.selectedLayerId
    );
    next[project.id] = reconciled;
    changed ||= reconciled !== byProject[project.id];
  }
  for (const id of reconciledDocuments.keys()) {
    if (!(id in next)) {
      reconciledDocuments.delete(id);
    }
  }
  if (changed || Object.keys(next).length !== Object.keys(byProject).length) {
    store.setSnapshot({ byProject: next });
  }
};

export const clearLayerPanelStates = (): void => {
  reconciledDocuments.clear();
  if (Object.keys(store.getSnapshot().byProject).length > 0) {
    store.setSnapshot({ byProject: {} });
  }
};

registerAccountOwnedResource({ clear: clearLayerPanelStates, name: 'layer-panel-state' });
