import type { CollisionDetection, DragEndEvent, DragMoveEvent, DragOverEvent, DragStartEvent } from '@dnd-kit/core';
import type { CanvasDocumentContractV3, DocumentCommand, LayerStackKind } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { LayerPanelState, LayerSelectionModifiers } from '@workbench/layerPanelState';
import type { Dispatch, FocusEvent, KeyboardEvent, ReactNode } from 'react';

import { Box, Text } from '@chakra-ui/react';
import { DndContext, DragOverlay, pointerWithin, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { restrictToVerticalAxis } from '@dnd-kit/modifiers';
import { useMountEffect } from '@platform/react/useMountEffect';
import { toaster } from '@platform/ui';
import { Scrollable } from '@platform/ui/Scrollable';
import { getDocumentIndex, getDocumentNode, lookupDocumentNodeState } from '@workbench/canvas-engine/api';
import {
  publishLayerPanelSelection,
  selectLayerInPanel,
  setLayerChildrenCollapsed,
  setLayerGroupExpanded,
  setLayerPanelFocus,
  toggleLayerStackCollapsed,
} from '@workbench/layerPanelState';
import { useCanvasOperation } from '@workbench/widgets/canvas/engineStoreHooks';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useEffectEvent, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { defaultRangeExtractor, useVirtualizer, type Range } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { LayerRowCommands, LayerSurfaceAnchor } from './layerRowCommands';
import type { LayerStackRowsByKind } from './layerTreeRows';
import type { LayerStackActionsEngine } from './useLayerStackActions';

import { ChildDragGhost, isRenameableChildKind, LayerChildRow } from './LayerChildRow';
import {
  layerChildDropCommand,
  layerChildRemoveLabelKey,
  layerChildRenameLabelKey,
  layerChildRowCommand,
  layerChildRowKey,
  projectLayerChildRows,
  type LayerChildDropTarget,
  type ProjectedChildRow,
} from './layerChildRows';
import { clearLayerChildSelection, selectLayerChild, useLayerChildSelection } from './layerChildSelection';
import { copyLayerToClipboard } from './layerExportActions';
import { createAdjustmentId } from './layerOps';
import {
  flattenPanelRows,
  isHeaderKey,
  isTreeNavigationKey,
  LAYER_HEADER_HEIGHT_PX,
  LAYER_TREE_INDENT_PX,
  navigateTree,
  panelRowHeight,
  stackOfHeaderKey,
  type PanelChildRowsSource,
  type PanelRow,
} from './layerPanelRows';
import { clearLayerPropertiesRequest, useCurrentLayerPropertiesRequest } from './layerPropertiesRequestStore';
import { LayerDragGhost, LayerRow, type LayerRowDragState } from './LayerRow';
import { anchorFromRect } from './layerRowCommands';
import { LayerStackHeader } from './LayerStackHeader';
import { LayerSurfaceHost, type LayerSurfaceEngine, type LayerSurfaceRequest } from './LayerSurfaceHost';
import { createLayerTreeAutoScroller, type LayerTreeAutoScroller } from './layerTreeAutoScroll';
import { outermostRowIds, projectLayerDrop, type LayerDropTarget } from './layerTreeDrop';

export type LayersTreeEngine = LayerStackActionsEngine & LayerSurfaceEngine & Pick<CanvasEngineHandle, 'previews'>;

const POINTER_SENSOR_OPTIONS = { activationConstraint: { distance: 6 } } as const;
const OVERLAY_MODIFIERS = [restrictToVerticalAxis];
/** How long the pointer rests on a collapsed group before it opens to accept the drop. */
const HOVER_EXPAND_DELAY_MS = 600;
const OVERSCAN_ROWS = 6;

interface DragState {
  readonly activeId: string;
  readonly stack: LayerStackKind;
  /** The outermost selected rows that travel, in document order. */
  readonly activeIds: readonly string[];
  readonly travelling: ReadonlySet<string>;
  /** How many selected rows travel, the number the ghost shows. */
  readonly selectedCount: number;
  readonly overId: string | null;
  readonly overIsGroup: boolean;
  /** The pointer sits on a projected child row; the drop is pinned below its owner. */
  readonly overIsChild: boolean;
  readonly edge: 'above' | 'inside' | 'below';
  readonly depthOffset: number;
}

interface ChildDragState {
  readonly child: ProjectedChildRow;
  readonly overKey: string | null;
  readonly edge: 'above' | 'below';
}

interface LayersTreeProps {
  degraded: boolean;
  dispatch: Dispatch<CanvasProjectMutation>;
  document: CanvasDocumentContractV3;
  editingLocked: boolean;
  engine: LayersTreeEngine | null;
  /** Reveals the Properties pane's Layer section for `layerId` (selecting it). */
  onRevealProperties: (layerId: string) => void;
  panel: LayerPanelState;
  projectId: string;
  stacks: LayerStackRowsByKind;
}

/** Which half of the row under the pointer it sits in. */
/** Use a group's middle drop band to enter it, including empty groups; leaves split at their midpoint. */
const edgeOf = (
  rect: { top: number; height: number } | undefined,
  y: number,
  overIsGroup: boolean
): 'above' | 'inside' | 'below' => {
  if (!rect) {
    return 'above';
  }
  if (overIsGroup) {
    const fraction = (y - rect.top) / rect.height;
    return fraction < 0.3 ? 'above' : fraction > 0.7 ? 'below' : 'inside';
  }
  return y >= rect.top + rect.height / 2 ? 'below' : 'above';
};

/** Which half of a child row the pointer sits in; landings are always between rows. */
const childEdgeOf = (rect: { top: number; height: number } | undefined, y: number): 'above' | 'below' =>
  !rect || y >= rect.top + rect.height / 2 ? 'below' : 'above';

type ReparentCommand = Extract<DocumentCommand, { type: 'reparent' }>;

const reparentCommandForTarget = (target: LayerDropTarget): ReparentCommand => ({
  beforeId: target.beforeId,
  ids: target.ids,
  parentId: target.parentId,
  type: 'reparent',
});

const isMenuKey = (event: KeyboardEvent<HTMLElement>): boolean =>
  event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey);

/** The viewport box of a rendered item, the anchor a keyboard-opened menu uses. */
const anchorOfItem = (host: HTMLElement | null, key: string): LayerSurfaceAnchor | null => {
  const element = host?.querySelector<HTMLElement>(`[data-layer-row-id="${CSS.escape(key)}"]`);
  return element ? anchorFromRect(element.getBoundingClientRect()) : null;
};

/** One absolutely positioned slot of the virtual list; the style object is built once per offset. */
const VirtualSlot = ({ children, size, start }: { children: ReactNode; size: number; start: number }) => {
  const style = useMemo(() => ({ height: size, top: start }), [size, start]);
  return (
    // Absolute children ignore the container's padding; the slot carries the
    // panel-edge inset itself, and the indicator and pinned header match it.
    <Box left="1.5" position="absolute" right="1.5" role="presentation" style={style}>
      {children}
    </Box>
  );
};

/**
 * Own subscriptions and commits in one virtualized tree with fixed rows, shared drag/menu hosts, and one roving
 * tab stop.
 */
export const LayersTree = ({
  degraded,
  dispatch,
  document,
  editingLocked,
  engine,
  onRevealProperties,
  panel,
  projectId,
  stacks,
}: LayersTreeProps) => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const propertiesRequest = useCurrentLayerPropertiesRequest();
  const operation = useCanvasOperation(engine);
  const operationLayerId = operation.status === 'active' ? operation.identity.layerId : null;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [surface, setSurface] = useState<LayerSurfaceRequest | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const dragRef = useRef<DragState | null>(null);
  useLayoutEffect(() => {
    dragRef.current = drag;
  }, [drag]);
  const [childDrag, setChildDrag] = useState<ChildDragState | null>(null);
  const childDragRef = useRef<ChildDragState | null>(null);
  useLayoutEffect(() => {
    childDragRef.current = childDrag;
  }, [childDrag]);
  const pointerStart = useRef({ x: 0, y: 0 });
  const pointerY = useRef(0);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingFocus = useRef<string | null>(null);
  const pendingProperties = useRef<{ id: string; token: number; scrolled: boolean; opened: boolean } | null>(null);
  // Whether keyboard focus lives in the tree; read by the repair effect after the focused row is gone.
  const treeOwnsFocus = useRef(false);
  // The primary the panel itself selected last; any other primary change came from outside and is revealed.
  const panelSelectedPrimary = useRef<string | null>(null);
  const revealedPrimary = useRef<string | null>(null);
  const { selectedIds, primaryId } = panel;

  const requestedId = propertiesRequest?.layerId ?? null;
  const requestedStack = useMemo(
    () => (requestedId ? (getDocumentIndex(document).byId.get(requestedId)?.stack ?? null) : null),
    [document, requestedId]
  );
  const forceOpen = useCallback((stack: LayerStackKind) => stack === requestedStack, [requestedStack]);
  const childRowsSource = useMemo<PanelChildRowsSource | undefined>(
    () =>
      panel.filter.trim() === ''
        ? { collapsedLayerIds: new Set(panel.collapsedChildLayerIds), rowsFor: (row) => projectLayerChildRows(row.vm) }
        : undefined,
    [panel.collapsedChildLayerIds, panel.filter]
  );
  const panelRows = useMemo(
    () => flattenPanelRows(stacks, panel.collapsedStacks, forceOpen, childRowsSource),
    [childRowsSource, forceOpen, panel.collapsedStacks, stacks]
  );
  const rowIndexByKey = useMemo(() => new Map(panelRows.map((row, index) => [row.key, index])), [panelRows]);
  const childRowByKey = useMemo(
    () => new Map(panelRows.flatMap((row) => (row.kind === 'child' ? [[row.key, row.child] as const] : []))),
    [panelRows]
  );
  const childSelection = useLayerChildSelection();
  const selectedChildKey =
    childSelection && childSelection.projectId === projectId
      ? layerChildRowKey(childSelection.layerId, childSelection.itemId)
      : null;
  const selectedChildOwnerId = childSelection && childSelection.projectId === projectId ? childSelection.layerId : null;
  const primaryAncestorIds = useMemo(
    () => new Set(primaryId ? (lookupDocumentNodeState(document, primaryId)?.parentIds ?? []) : []),
    [document, primaryId]
  );
  const offsets = useMemo(() => {
    const starts = new Float64Array(panelRows.length + 1);
    panelRows.forEach((row, index) => {
      starts[index + 1] = starts[index]! + panelRowHeight(row);
    });
    return starts;
  }, [panelRows]);
  const headerIndexes = useMemo(
    () => panelRows.flatMap((row, index) => (row.kind === 'header' ? [index] : [])),
    [panelRows]
  );
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const allNodeIds = useMemo(() => Object.values(stacks).flatMap((stack) => stack.nodeIds), [stacks]);
  const visibleRowIds = useMemo(
    () => panelRows.flatMap((row) => (row.kind === 'node' ? [row.row.id] : [])),
    [panelRows]
  );
  const focusKey =
    panel.focusId && rowIndexByKey.has(panel.focusId)
      ? panel.focusId
      : primaryId && rowIndexByKey.has(primaryId)
        ? primaryId
        : (panelRows[0]?.key ?? null);
  const focusIndex = focusKey ? (rowIndexByKey.get(focusKey) ?? -1) : -1;

  // The focused item stays mounted whatever the scroll position: focus must never fall off the tree.
  const rangeExtractor = useCallback(
    (range: Range) => {
      const indexes = defaultRangeExtractor(range);
      if (focusIndex >= 0 && !indexes.includes(focusIndex)) {
        indexes.push(focusIndex);
        indexes.sort((a, b) => a - b);
      }
      return indexes;
    },
    [focusIndex]
  );
  const getScrollElement = useCallback(() => scrollRef.current, []);
  const autoScroller = useRef<LayerTreeAutoScroller | null>(null);
  const estimateSize = useCallback((index: number) => panelRowHeight(panelRows[index]!), [panelRows]);
  const getItemKey = useCallback((index: number) => panelRows[index]?.key ?? index, [panelRows]);
  const virtualizer = useVirtualizer({
    count: panelRows.length,
    estimateSize,
    getItemKey,
    getScrollElement,
    overscan: OVERSCAN_ROWS,
    rangeExtractor,
    scrollPaddingStart: LAYER_HEADER_HEIGHT_PX,
  });
  const virtualItems = virtualizer.virtualItems;
  const measureVirtualizer = useEffectEvent(() => {
    virtualizer.measure();
  });
  useLayoutEffect(() => {
    measureVirtualizer();
  }, [panelRows]);
  const virtualizerRef = useRef(virtualizer);
  useLayoutEffect(() => {
    virtualizerRef.current = virtualizer;
  });

  const handleScroll = useCallback(() => setScrollTop(scrollRef.current?.scrollTop ?? 0), []);

  // Event handlers read the latest render through this ref, so the command handle never changes identity.
  const latest = useRef({
    allNodeIds,
    childRowByKey,
    commitPrepared,
    dispatch,
    document,
    editingLocked,
    engine,
    onRevealProperties,
    panel,
    panelRows,
    projectId,
    rowIndexByKey,
    stacks,
    visibleRowIds,
  });
  useLayoutEffect(() => {
    latest.current = {
      allNodeIds,
      childRowByKey,
      commitPrepared,
      dispatch,
      document,
      editingLocked,
      engine,
      onRevealProperties,
      panel,
      panelRows,
      projectId,
      rowIndexByKey,
      stacks,
      visibleRowIds,
    };
  });
  const movingIds = useCallback((key: string, stack: LayerStackKind): readonly string[] => {
    const { panel: current, stacks: forests } = latest.current;
    return current.selectedIds.includes(key)
      ? outermostRowIds(forests[stack].rows, new Set(current.selectedIds))
      : [key];
  }, []);

  const focusItem = useCallback((key: string) => {
    const { panel: current, projectId: project, rowIndexByKey: keys } = latest.current;
    setLayerPanelFocus(project, current.primaryId, key);
    const index = keys.get(key);
    if (index !== undefined) {
      virtualizerRef.current.scrollToIndex(index);
    }
    pendingFocus.current = key;
  }, []);

  const runStructural = useCallback((label: string, command: DocumentCommand) => {
    const { commitPrepared: commit } = latest.current;
    return commit(label, (model) => model.prepare(command));
  }, []);
  const removeChildRow = useCallback(
    (child: ProjectedChildRow) => {
      const command = layerChildRowCommand(latest.current.document, child, { type: 'remove' });
      if (!command) {
        return;
      }
      const { panel: current, projectId: project } = latest.current;
      const outcome = runStructural(t(layerChildRemoveLabelKey(child.kind)), command);
      if (outcome.status === 'committed' && current.focusId === child.key) {
        setLayerPanelFocus(project, current.primaryId, child.layerId);
      }
    },
    [runStructural, t]
  );

  // After deletion, focus the nearest surviving row below, otherwise above.
  const removeRows = useCallback(
    (key: string) => {
      const { document: currentDocument, engine: currentEngine, panel: current, visibleRowIds: rows } = latest.current;
      const vm = lookupDocumentNodeState(currentDocument, key);
      if (!vm) {
        return;
      }
      // The menu's rule: the selection when the row is part of it, else the row alone.
      const ids = current.selectedIds.includes(key) ? current.selectedIds : [key];
      // A stray key on a locked row is ignored, as the menu disables its Delete.
      if (currentEngine?.document.model()?.refusalFor({ ids, type: 'remove' })) {
        return;
      }
      const removed = new Set(ids);
      const survives = (id: string): boolean => {
        const state = lookupDocumentNodeState(currentDocument, id);
        return !!state && !removed.has(id) && !state.parentIds.some((ancestor) => removed.has(ancestor));
      };
      const index = rows.indexOf(key);
      const next = rows.slice(index + 1).find(survives) ?? rows.slice(0, Math.max(0, index)).reverse().find(survives);
      const outcome = runStructural(
        t(ids.length > 1 ? 'widgets.layers.actions.deleteSelected' : 'widgets.layers.actions.delete'),
        { ids, type: 'remove' }
      );
      if (outcome.status === 'committed' && next) {
        focusItem(next);
      }
    },
    [focusItem, runStructural, t]
  );

  const commands = useMemo<LayerRowCommands>(
    () => ({
      endRename: () => setRenamingId(null),
      focus: (key) => {
        const { panel: current, projectId: project } = latest.current;
        if (current.focusId !== key) {
          setLayerPanelFocus(project, current.primaryId, key);
        }
      },
      keyDown: (key, event: KeyboardEvent<HTMLElement>) => {
        const {
          childRowByKey: childRows,
          document: currentDocument,
          panel: current,
          panelRows: rows,
          projectId: project,
        } = latest.current;
        const childRow = childRows.get(key);
        if (isMenuKey(event)) {
          event.preventDefault();
          const anchor = anchorOfItem(scrollRef.current, key);
          if (anchor) {
            setSurface(
              childRow
                ? { anchor, child: childRow, kind: 'child-menu' }
                : isHeaderKey(key)
                  ? { anchor, kind: 'stack-menu', stack: stackOfHeaderKey(key) }
                  : { anchor, id: key, kind: 'menu' }
            );
          }
          return;
        }
        if (
          !childRow &&
          !isHeaderKey(key) &&
          (event.ctrlKey || event.metaKey) &&
          !event.shiftKey &&
          !event.altKey &&
          event.key.toLowerCase() === 'c'
        ) {
          event.preventDefault();
          const { engine: currentEngine } = latest.current;
          if (currentEngine) {
            void copyLayerToClipboard(key, { exportLayer: currentEngine.exports.exportBakedLayerBlob })
              .then((status) => status === 'ok' || Promise.reject(new Error(status)))
              .catch(() => toaster.create({ title: t('widgets.layers.actions.copyFailed'), type: 'warning' }));
          }
          return;
        }
        if (childRow && (event.key === 'Delete' || event.key === 'Backspace')) {
          event.preventDefault();
          if (!latest.current.editingLocked) {
            removeChildRow(childRow);
          }
          return;
        }
        if (!isHeaderKey(key) && (event.key === 'Delete' || event.key === 'Backspace')) {
          event.preventDefault();
          if (!latest.current.editingLocked) {
            removeRows(key);
          }
          return;
        }
        if (isHeaderKey(key)) {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            toggleLayerStackCollapsed(project, current.primaryId, stackOfHeaderKey(key));
            return;
          }
        } else if (event.altKey && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
          event.preventDefault();
          const vm = lookupDocumentNodeState(currentDocument, key);
          if (!vm) {
            return;
          }
          runStructural(t('widgets.layers.actions.reorder'), {
            ids: movingIds(key, vm.stack),
            kind: event.key === 'ArrowUp' ? 'forward' : 'backward',
            type: 'move',
          });
          return;
        } else if (event.altKey && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
          event.preventDefault();
          const vm = lookupDocumentNodeState(currentDocument, key);
          if (!vm) {
            return;
          }
          // Only siblings of the focused row can follow it into or out of a group.
          const ids = movingIds(key, vm.stack).filter(
            (id) => lookupDocumentNodeState(currentDocument, id)?.parentId === vm.parentId
          );
          if (ids.length === 0) {
            return;
          }
          if (event.key === 'ArrowLeft') {
            if (vm.parentId === null) {
              return;
            }
            const parent = lookupDocumentNodeState(currentDocument, vm.parentId)!;
            const grandSiblings =
              parent.parentId === null
                ? currentDocument.stacks[parent.stack]
                : (getDocumentNode(currentDocument, parent.parentId) as { children: readonly { id: string }[] })
                    .children;
            const beforeId = grandSiblings[parent.siblingIndex + 1]?.id ?? null;
            runStructural(t('widgets.layers.actions.outdent'), {
              beforeId,
              ids,
              parentId: parent.parentId,
              type: 'reparent',
            });
            return;
          }
          const siblings =
            vm.parentId === null
              ? currentDocument.stacks[vm.stack]
              : (getDocumentNode(currentDocument, vm.parentId) as { children: readonly { id: string; type: string }[] })
                  .children;
          const above = siblings[vm.siblingIndex - 1];
          if (!above || above.type !== 'group') {
            return;
          }
          const outcome = runStructural(t('widgets.layers.actions.indent'), {
            beforeId: null,
            ids,
            parentId: above.id,
            type: 'reparent',
          });
          if (outcome.status === 'committed') {
            setLayerGroupExpanded(project, current.primaryId, [above.id], true);
          }
          return;
        } else if (event.key === 'F2') {
          event.preventDefault();
          if (!latest.current.editingLocked && (!childRow || isRenameableChildKind(childRow.kind))) {
            setRenamingId(key);
          }
          return;
        }
        if (!isTreeNavigationKey(event.key)) {
          return;
        }
        const navigation = navigateTree(rows, key, event.key);
        if (!navigation) {
          return;
        }
        event.preventDefault();
        if ('expand' in navigation) {
          setLayerGroupExpanded(project, current.primaryId, [navigation.expand], navigation.expanded);
          return;
        }
        if ('expandChildren' in navigation) {
          setLayerChildrenCollapsed(project, current.primaryId, navigation.expandChildren, !navigation.expanded);
          return;
        }
        if ('collapseStack' in navigation) {
          if (current.collapsedStacks.includes(navigation.collapseStack) !== navigation.collapsed) {
            toggleLayerStackCollapsed(project, current.primaryId, navigation.collapseStack);
          }
          return;
        }
        focusItem(navigation.focus);
      },
      duplicateChild: (child) => {
        const command = layerChildRowCommand(latest.current.document, child, {
          newId: createAdjustmentId(),
          type: 'duplicate',
        });
        if (command) {
          runStructural(t('widgets.layers.modifiers.duplicateAdjustment'), command);
        }
      },
      moveChild: (child, direction) => {
        const command = layerChildRowCommand(latest.current.document, child, { direction, type: 'move' });
        if (command) {
          runStructural(t('widgets.layers.modifiers.reorderAdjustment'), command);
        }
      },
      moveChildToLayer: (child, layerId) => {
        const command = layerChildDropCommand(latest.current.document, child, { beforeItemId: null, layerId });
        if (command) {
          runStructural(t('widgets.layers.modifiers.moveReferenceImage'), command);
        }
      },
      openChildMenu: (child, anchor: LayerSurfaceAnchor) => setSurface({ anchor, child, kind: 'child-menu' }),
      openMenu: (id, anchor: LayerSurfaceAnchor) => setSurface({ anchor, id, kind: 'menu' }),
      openStackMenu: (stack, anchor: LayerSurfaceAnchor) => setSurface({ anchor, kind: 'stack-menu', stack }),
      removeChild: (child) => removeChildRow(child),
      rename: (id, name) => runStructural(t('widgets.layers.actions.rename'), { id, patch: { name }, type: 'patch' }),
      renameChild: (child, name) => {
        const command = layerChildRowCommand(latest.current.document, child, { name, type: 'rename' });
        if (command) {
          runStructural(t(layerChildRenameLabelKey(child.kind)), command);
        }
      },
      select: (id, modifiers: LayerSelectionModifiers) => {
        const {
          allNodeIds: all,
          dispatch: send,
          panel: current,
          projectId: project,
          visibleRowIds: visible,
        } = latest.current;
        clearLayerChildSelection();
        const next = selectLayerInPanel(current, id, modifiers.range ? visible : all, modifiers);
        publishLayerPanelSelection(next);
        setLayerPanelFocus(project, next.primaryId, id);
        panelSelectedPrimary.current = next.primaryId;
        if (next.primaryId !== current.primaryId) {
          send({ id: next.primaryId, type: 'setCanvasSelectedLayer' });
        }
      },
      selectChild: (child, options) => {
        const { dispatch: send, panel: current, projectId: project } = latest.current;
        // The owner becomes the single selected layer, published before the
        // dispatch so reconciliation keeps it.
        publishLayerPanelSelection({ primaryId: child.layerId, projectId: project, selectedIds: [child.layerId] });
        panelSelectedPrimary.current = child.layerId;
        if (current.primaryId !== child.layerId) {
          send({ id: child.layerId, type: 'setCanvasSelectedLayer' });
        }
        selectLayerChild(project, child.layerId, child.itemId);
        setLayerPanelFocus(project, child.layerId, child.key);
        if (options?.reveal !== false) {
          latest.current.onRevealProperties(child.layerId);
        }
      },
      setChildEnabled: (child, isEnabled) => {
        const command = layerChildRowCommand(latest.current.document, child, { isEnabled, type: 'set-enabled' });
        if (command) {
          runStructural(t('widgets.layers.modifiers.toggleActive'), command);
        }
      },
      setEnabled: (id, isEnabled) =>
        runStructural(t('widgets.layers.actions.toggleActive'), {
          type: 'set-enabled',
          updates: [{ id, isEnabled }],
        }),
      setHidden: (id, isHidden) =>
        runStructural(t('widgets.layers.actions.toggleHidden'), { type: 'set-hidden', updates: [{ id, isHidden }] }),
      setLocked: (id, isLocked) =>
        runStructural(t('widgets.layers.actions.toggleLock'), { type: 'set-locked', updates: [{ id, isLocked }] }),
      startRename: (id) => setRenamingId(id),
      toggleChildren: (id) => {
        const { panel: current, projectId: project } = latest.current;
        setLayerChildrenCollapsed(project, current.primaryId, id);
      },
      toggleCollapse: (stack) => {
        const { panel: current, projectId: project } = latest.current;
        toggleLayerStackCollapsed(project, current.primaryId, stack);
      },
      toggleExpanded: (id) => {
        const { panel: current, projectId: project } = latest.current;
        if (current.filter.trim() === '') {
          setLayerGroupExpanded(project, current.primaryId, [id]);
        }
      },
    }),
    [focusItem, movingIds, removeChildRow, removeRows, runStructural, t]
  );

  const closeSurface = useCallback(() => {
    setSurface(null);
    if (pendingProperties.current) {
      clearLayerPropertiesRequest(pendingProperties.current.token);
      pendingProperties.current = null;
    }
  }, []);

  // A primary that changed outside the panel (canvas, undo, a new layer) is scrolled into view once,
  // as soon as it has a row; later list changes leave the scroll position alone.
  useLayoutEffect(() => {
    if (!primaryId || revealedPrimary.current === primaryId) {
      return;
    }
    if (panelSelectedPrimary.current === primaryId) {
      panelSelectedPrimary.current = null;
      revealedPrimary.current = primaryId;
      return;
    }
    const index = rowIndexByKey.get(primaryId);
    if (index !== undefined) {
      revealedPrimary.current = primaryId;
      virtualizerRef.current.scrollToIndex(index);
    }
  }, [primaryId, rowIndexByKey]);

  // Reveal external Properties requests, scroll after render, and open once; discard requests for removed nodes
  // and, while an operation runs, for any layer but its target: the operation's panel supersedes them.
  useLayoutEffect(() => {
    if (!propertiesRequest) {
      pendingProperties.current = null;
      return;
    }
    if (
      !getDocumentNode(document, propertiesRequest.layerId) ||
      (operationLayerId !== null && operationLayerId !== propertiesRequest.layerId)
    ) {
      clearLayerPropertiesRequest(propertiesRequest.token);
      pendingProperties.current = null;
      return;
    }
    let pending = pendingProperties.current;
    if (pending?.token !== propertiesRequest.token) {
      pending = { id: propertiesRequest.layerId, opened: false, scrolled: false, token: propertiesRequest.token };
      pendingProperties.current = pending;
      const vm = lookupDocumentNodeState(document, propertiesRequest.layerId);
      if (vm && vm.parentIds.length > 0) {
        setLayerGroupExpanded(projectId, primaryId, [...vm.parentIds], true);
      }
    }
    const index = rowIndexByKey.get(pending.id);
    if (!pending.scrolled && index !== undefined) {
      pendingProperties.current = { ...pending, scrolled: true };
      virtualizerRef.current.scrollToIndex(index);
    }
  }, [document, operationLayerId, primaryId, projectId, propertiesRequest, rowIndexByKey]);

  useLayoutEffect(() => {
    const host = scrollRef.current;
    if (!host) {
      return;
    }
    if (pendingFocus.current) {
      const target = host.querySelector<HTMLElement>(`[data-layer-row-id="${CSS.escape(pendingFocus.current)}"]`);
      if (target) {
        target.focus();
        pendingFocus.current = null;
      }
    }
    const pending = pendingProperties.current;
    if (pending && pending.scrolled && !pending.opened) {
      const row = host.querySelector<HTMLElement>(`[data-layer-row-id="${CSS.escape(pending.id)}"]`);
      if (row) {
        pendingProperties.current = null;
        onRevealProperties(pending.id);
        clearLayerPropertiesRequest(pending.token);
      }
    }
  }, [onRevealProperties, panelRows, virtualItems]);

  // Repair focus after row unmount only if the actual tree DOM owned it; portaled focus does not count.
  const handleFocusCapture = useCallback((event: FocusEvent<HTMLElement>) => {
    if (event.currentTarget.contains(event.target as Node)) {
      treeOwnsFocus.current = true;
    }
  }, []);
  const handleBlurCapture = useCallback((event: FocusEvent<HTMLElement>) => {
    if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) {
      treeOwnsFocus.current = false;
    }
  }, []);
  const scrollViewportProps = useMemo(
    () => ({ onBlurCapture: handleBlurCapture, onFocusCapture: handleFocusCapture, onScroll: handleScroll }),
    [handleBlurCapture, handleFocusCapture, handleScroll]
  );
  useLayoutEffect(() => {
    const host = scrollRef.current;
    if (!host || !focusKey || !treeOwnsFocus.current || renamingId !== null) {
      return;
    }
    if (host.contains(window.document.activeElement) && window.document.activeElement !== window.document.body) {
      return;
    }
    host.querySelector<HTMLElement>(`[data-layer-row-id="${CSS.escape(focusKey)}"]`)?.focus();
  }, [focusKey, panelRows, renamingId, virtualItems]);

  const clearHoverTimer = useCallback(() => {
    if (hoverTimer.current !== null) {
      clearTimeout(hoverTimer.current);
      hoverTimer.current = null;
    }
  }, []);
  useMountEffect(() => {
    autoScroller.current = createLayerTreeAutoScroller(scrollRef);
    return () => {
      clearHoverTimer();
      autoScroller.current?.stop();
    };
  });

  const sensors = useSensors(useSensor(PointerSensor, POINTER_SENSOR_OPTIONS));
  const dragDisabled = editingLocked || degraded || panel.filter.trim() !== '';
  const collisionDetection = useCallback<CollisionDetection>((args) => {
    const stack = args.active.data.current?.stack as LayerStackKind | undefined;
    const { childRowByKey: childRows } = latest.current;
    const dragChild = childRows.get(String(args.active.id));
    if (dragChild) {
      return pointerWithin({
        ...args,
        droppableContainers: args.droppableContainers.filter((container) => {
          const id = String(container.id);
          if (id === dragChild.key) {
            return false;
          }
          const overChild = childRows.get(id);
          if (dragChild.kind === 'reference-image') {
            return overChild
              ? overChild.kind === 'reference-image'
              : container.data.current?.stack === 'regional_guidance';
          }
          return overChild ? overChild.layerId === dragChild.layerId : id === dragChild.layerId;
        }),
      });
    }
    const current = dragRef.current;
    return pointerWithin({
      ...args,
      droppableContainers: args.droppableContainers.filter((container) => {
        if (container.data.current?.stack !== stack) {
          return false;
        }
        const id = String(container.id);
        const owner = childRows.get(id)?.layerId ?? id;
        return !(current?.travelling.has(owner) ?? false);
      }),
    });
  }, []);

  const handleDragStart = useCallback(
    (event: DragStartEvent) => {
      const activeId = String(event.active.id);
      const activator = event.activatorEvent as PointerEvent;
      const dragChild = childRowByKey.get(activeId);
      if (dragChild) {
        pointerStart.current = { x: activator.clientX ?? 0, y: activator.clientY ?? 0 };
        pointerY.current = pointerStart.current.y;
        setChildDrag({ child: dragChild, edge: 'above', overKey: null });
        return;
      }
      const stack = event.active.data.current?.stack as LayerStackKind;
      const rows = stacks[stack].rows;
      const selected = new Set(selectedSet.has(activeId) ? selectedIds : [activeId]);
      const activeIds = outermostRowIds(rows, selected);
      const travelling = new Set<string>();
      const active = new Set(activeIds);
      let covering: number | null = null;
      for (const row of rows) {
        if (covering !== null && row.vm.depth > covering) {
          travelling.add(row.id);
          continue;
        }
        covering = null;
        if (active.has(row.id)) {
          travelling.add(row.id);
          covering = row.vm.depth;
        }
      }
      pointerStart.current = { x: activator.clientX ?? 0, y: activator.clientY ?? 0 };
      pointerY.current = pointerStart.current.y;
      setDrag({
        activeId,
        activeIds,
        depthOffset: 0,
        edge: 'above',
        overId: null,
        overIsChild: false,
        overIsGroup: false,
        selectedCount: [...travelling].filter((id) => selected.has(id)).length,
        stack,
        travelling,
      });
    },
    [childRowByKey, selectedIds, selectedSet, stacks]
  );
  const handleDragMove = useCallback((event: DragMoveEvent) => {
    pointerY.current = pointerStart.current.y + event.delta.y;
    autoScroller.current?.update(pointerY.current);
    if (childDragRef.current) {
      setChildDrag((current) => {
        if (!current) {
          return current;
        }
        const edge = childEdgeOf(event.over?.rect, pointerY.current);
        return current.edge !== edge ? { ...current, edge } : current;
      });
      return;
    }
    const depthOffset = Math.round(event.delta.x / LAYER_TREE_INDENT_PX);
    setDrag((current) => {
      if (!current) {
        return current;
      }
      const edge = current.overIsChild ? 'below' : edgeOf(event.over?.rect, pointerY.current, current.overIsGroup);
      return current.depthOffset !== depthOffset || current.edge !== edge ? { ...current, depthOffset, edge } : current;
    });
  }, []);
  const handleDragOver = useCallback(
    (event: DragOverEvent) => {
      const rawOverId = event.over ? String(event.over.id) : null;
      if (childDragRef.current) {
        const edge = childEdgeOf(event.over?.rect, pointerY.current);
        setChildDrag((state) =>
          state && (state.overKey !== rawOverId || state.edge !== edge) ? { ...state, edge, overKey: rawOverId } : state
        );
        return;
      }
      const overChild = rawOverId ? childRowByKey.get(rawOverId) : undefined;
      const overId = overChild ? overChild.layerId : rawOverId;
      clearHoverTimer();
      const current = dragRef.current;
      const over =
        overId && current && overId !== current.activeId
          ? stacks[current.stack].rows.find((row) => row.id === overId)
          : undefined;
      const overIsGroup = over?.vm.kind === 'group';
      const edge = overChild ? 'below' : edgeOf(event.over?.rect, pointerY.current, overIsGroup);
      const overIsChild = overChild !== undefined;
      setDrag((state) =>
        state &&
        (state.overId !== overId ||
          state.edge !== edge ||
          state.overIsGroup !== overIsGroup ||
          state.overIsChild !== overIsChild)
          ? { ...state, edge, overId, overIsChild, overIsGroup }
          : state
      );
      if (over && over.vm.kind === 'group' && !over.expanded && over.vm.childCount > 0) {
        hoverTimer.current = setTimeout(
          () => setLayerGroupExpanded(projectId, primaryId, [over.id], true),
          HOVER_EXPAND_DELAY_MS
        );
      }
    },
    [childRowByKey, clearHoverTimer, primaryId, projectId, stacks]
  );

  const target = useMemo((): LayerDropTarget | null => {
    if (!drag || !drag.overId) {
      return null;
    }
    return projectLayerDrop({
      activeIds: drag.activeIds,
      depthOffset: drag.depthOffset,
      edge: drag.edge,
      overId: drag.overId,
      rows: stacks[drag.stack].rows,
    });
  }, [drag, stacks]);
  // Cache drag validation by semantic move rather than equivalent visual gaps; clear on completion and revalidate
  // the prepared landing against the latest document.
  const refusalCommandKey = target ? JSON.stringify(reparentCommandForTarget(target)) : null;
  const refusal = useMemo(
    () =>
      refusalCommandKey && engine
        ? (engine.document.model()?.refusalFor(JSON.parse(refusalCommandKey) as ReparentCommand) ?? null)
        : null,
    [engine, refusalCommandKey]
  );

  const childDropPlan = useMemo(() => {
    if (!childDrag?.overKey) {
      return null;
    }
    const { child, edge, overKey } = childDrag;
    const overChild = childRowByKey.get(overKey);
    let dropTarget: LayerChildDropTarget;
    let depth: number;
    if (overChild) {
      const ownerRow = stacks[overChild.stack].rows.find((row) => row.id === overChild.layerId);
      const siblings = ownerRow ? projectLayerChildRows(ownerRow.vm) : [];
      const index = siblings.findIndex((sibling) => sibling.itemId === overChild.itemId);
      dropTarget = {
        beforeItemId: edge === 'above' ? overChild.itemId : (siblings[index + 1]?.itemId ?? null),
        layerId: overChild.layerId,
      };
      depth = overChild.depth;
    } else {
      const ownerRow = stacks[child.stack].rows.find((row) => row.id === overKey);
      if (!ownerRow) {
        return null;
      }
      dropTarget = { beforeItemId: null, layerId: overKey };
      depth = ownerRow.vm.depth + 1;
    }
    const command = layerChildDropCommand(document, child, dropTarget);
    if (!command) {
      return null;
    }
    if (engine && engine.document.model()?.refusalFor(command)) {
      return null;
    }
    return { command, depth, dropTarget };
  }, [childDrag, childRowByKey, document, engine, stacks]);

  const childIndicator = useMemo(() => {
    if (!childDropPlan) {
      return null;
    }
    const { depth, dropTarget } = childDropPlan;
    const beforeKey = dropTarget.beforeItemId ? layerChildRowKey(dropTarget.layerId, dropTarget.beforeItemId) : null;
    const beforeIndex = beforeKey ? rowIndexByKey.get(beforeKey) : undefined;
    if (beforeIndex !== undefined) {
      return { left: depth * LAYER_TREE_INDENT_PX + 14, top: offsets[beforeIndex]! };
    }
    let index = rowIndexByKey.get(dropTarget.layerId);
    if (index === undefined) {
      return null;
    }
    while (panelRows[index + 1]?.kind === 'child') {
      index += 1;
    }
    return { left: depth * LAYER_TREE_INDENT_PX + 14, top: offsets[index + 1]! };
  }, [childDropPlan, offsets, panelRows, rowIndexByKey]);

  const finishDrag = useCallback(() => {
    clearHoverTimer();
    autoScroller.current?.stop();
    setDrag(null);
    setChildDrag(null);
  }, [clearHoverTimer]);
  const handleDragEnd = useCallback(
    (_event: DragEndEvent) => {
      const landing = target;
      const refused = refusal;
      const childLanding = childDragRef.current ? childDropPlan : null;
      const childKind = childDragRef.current?.child.kind;
      finishDrag();
      if (dragDisabled) {
        return;
      }
      if (childKind) {
        if (childLanding) {
          const label =
            childKind === 'reference-image'
              ? t('widgets.layers.modifiers.moveReferenceImage')
              : t('widgets.layers.modifiers.reorderAdjustment');
          commitPrepared(label, (model) => model.prepare(childLanding.command));
        }
        return;
      }
      if (!landing || refused) {
        return;
      }
      commitPrepared(t('widgets.layers.actions.reorder'), (model) =>
        model.prepare({ beforeId: landing.beforeId, ids: landing.ids, parentId: landing.parentId, type: 'reparent' })
      );
    },
    [childDropPlan, commitPrepared, dragDisabled, finishDrag, refusal, t, target]
  );

  const indicator = useMemo(() => {
    if (!target || !drag) {
      return null;
    }
    const stackRows = stacks[drag.stack].rows;
    let top: number;
    if (target.beforeRowId) {
      top = offsets[rowIndexByKey.get(target.beforeRowId)!]!;
    } else {
      const last = stackRows[stackRows.length - 1]!;
      let index = rowIndexByKey.get(last.id)!;
      while (panelRows[index + 1]?.kind === 'child') {
        index += 1;
      }
      top = offsets[index + 1]!;
    }
    return { left: target.depth * LAYER_TREE_INDENT_PX + 14, top };
  }, [drag, offsets, panelRows, rowIndexByKey, stacks, target]);

  const refusalReason = refusal
    ? t(
        refusal.status === 'locked'
          ? 'widgets.canvas.structural.refusedLocked'
          : refusal.status === 'invalid-target' && refusal.reason === 'cycle'
            ? 'widgets.canvas.structural.refusedCycle'
            : refusal.status === 'invalid-target' && refusal.reason === 'depth-exceeded'
              ? 'widgets.canvas.structural.refusedDepth'
              : 'widgets.canvas.structural.refusedInvalidTarget'
      )
    : null;

  // The stack whose header scrolled out keeps a pinned copy; the next header pushes it up as it arrives.
  const pinnedHeader = useMemo(() => {
    let low = 0;
    let high = headerIndexes.length - 1;
    let pinned = -1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      if (offsets[headerIndexes[middle]!]! < scrollTop) {
        pinned = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    if (pinned < 0) {
      return null;
    }
    const next = headerIndexes[pinned + 1];
    const nextTop = next === undefined ? Number.POSITIVE_INFINITY : offsets[next]! - scrollTop;
    return {
      row: panelRows[headerIndexes[pinned]!] as Extract<PanelRow, { kind: 'header' }>,
      shift: Math.min(0, nextTop - LAYER_HEADER_HEIGHT_PX),
    };
  }, [headerIndexes, offsets, panelRows, scrollTop]);
  const pinnedStyle = useMemo(
    () => (pinnedHeader ? { transform: `translateY(${pinnedHeader.shift}px)` } : undefined),
    [pinnedHeader]
  );

  const activeVm = drag ? lookupDocumentNodeState(document, drag.activeId) : null;
  const dragStateOf = (id: string): LayerRowDragState =>
    drag ? (id === drag.activeId ? 'source' : drag.travelling.has(id) ? 'travelling' : null) : null;

  const renderHeader = (row: Extract<PanelRow, { kind: 'header' }>, pinned: boolean) => (
    <LayerStackHeader
      collapsed={row.collapsed}
      commands={commands}
      document={document}
      editingLocked={editingLocked}
      engine={engine}
      focused={row.key === focusKey}
      leafCount={row.stack.leafCount}
      pinned={pinned}
      posInSet={row.posInSet}
      rowKey={row.key}
      setSize={row.setSize}
      stack={row.stack.stack}
    />
  );

  return (
    <DndContext
      autoScroll={false}
      collisionDetection={collisionDetection}
      sensors={sensors}
      onDragCancel={finishDrag}
      onDragEnd={handleDragEnd}
      onDragMove={handleDragMove}
      onDragOver={handleDragOver}
      onDragStart={handleDragStart}
    >
      {/* `overflow=hidden` clips the pinned header's push-out translate at the tree's edge. */}
      <Box flex="1" minH="0" overflow="hidden" position="relative">
        <Scrollable h="full" viewportProps={scrollViewportProps} viewportRef={scrollRef}>
          <Box
            aria-label={t('widgets.layers.tree')}
            aria-multiselectable="true"
            h={`${virtualizer.totalSize}px`}
            position="relative"
            role="tree"
            userSelect="none"
            w="full"
          >
            {virtualItems.map((item) => {
              // The window applies a changed key set one commit late; a slot draws the row its key
              // names, never the row at its index, so no node is relabelled in between.
              const rowIndex = rowIndexByKey.get(String(item.key));
              const panelRow = rowIndex === undefined ? undefined : panelRows[rowIndex];
              if (!panelRow) {
                return null;
              }
              return (
                <VirtualSlot key={item.key} size={item.size} start={item.start}>
                  {panelRow.kind === 'node' ? (
                    <LayerRow
                      ancestorOfPrimary={primaryAncestorIds.has(panelRow.row.id)}
                      ariaPosInSet={panelRow.ariaPosInSet}
                      ariaSetSize={panelRow.ariaSetSize}
                      childCount={panelRow.childCount}
                      childSelected={panelRow.row.id === selectedChildOwnerId}
                      childrenExpanded={panelRow.childrenExpanded}
                      commands={commands}
                      drag={dragStateOf(panelRow.row.id)}
                      dragDisabled={dragDisabled}
                      editingLocked={editingLocked}
                      engine={engine}
                      focused={panelRow.row.id === focusKey}
                      primary={panelRow.row.id === primaryId}
                      renaming={panelRow.row.id === renamingId}
                      row={panelRow.row}
                      selected={selectedSet.has(panelRow.row.id)}
                      thumbnails={!degraded}
                    />
                  ) : panelRow.kind === 'child' ? (
                    <LayerChildRow
                      ariaSetSize={panelRow.ariaSetSize}
                      child={panelRow.child}
                      commands={commands}
                      dimmed={
                        (drag?.travelling.has(panelRow.child.layerId) ?? false) || childDrag?.child.key === panelRow.key
                      }
                      dragDisabled={dragDisabled}
                      editingLocked={editingLocked}
                      focused={panelRow.key === focusKey}
                      renaming={panelRow.key === renamingId}
                      selected={panelRow.key === selectedChildKey}
                    />
                  ) : (
                    renderHeader(panelRow, false)
                  )}
                </VirtualSlot>
              );
            })}
            {childIndicator ? (
              <Box
                aria-hidden
                bg="accent.solid"
                h="2px"
                left={`${childIndicator.left}px`}
                pointerEvents="none"
                position="absolute"
                right="2"
                rounded="full"
                top={`${childIndicator.top - 1}px`}
              />
            ) : null}
            {indicator ? (
              <Box
                aria-hidden
                bg={refusal ? 'fg.error' : 'accent.solid'}
                h="2px"
                left={`${indicator.left}px`}
                pointerEvents="none"
                position="absolute"
                right="2"
                rounded="full"
                top={`${indicator.top - 1}px`}
              >
                {refusalReason ? (
                  <Text
                    bg="bg.panel"
                    color="fg.error"
                    fontSize="xs"
                    left="0"
                    maxW="full"
                    position="absolute"
                    px="1"
                    rounded="sm"
                    top="1"
                  >
                    {t('widgets.layers.dropRefused', { reason: refusalReason })}
                  </Text>
                ) : null}
              </Box>
            ) : null}
          </Box>
        </Scrollable>
        {pinnedHeader ? (
          <Box
            borderBottomWidth="1px"
            borderColor="border.subtle"
            h={`${LAYER_HEADER_HEIGHT_PX}px`}
            left="1.5"
            position="absolute"
            right="1.5"
            style={pinnedStyle}
            top="0"
            zIndex="1"
          >
            {renderHeader(pinnedHeader.row, true)}
          </Box>
        ) : null}
        <DragOverlay dropAnimation={null} modifiers={OVERLAY_MODIFIERS}>
          {childDrag ? (
            <ChildDragGhost child={childDrag.child} />
          ) : activeVm && drag ? (
            <LayerDragGhost count={drag.selectedCount} vm={activeVm} />
          ) : null}
        </DragOverlay>
      </Box>
      <LayerSurfaceHost
        commands={commands}
        dispatch={dispatch}
        document={document}
        editingLocked={editingLocked}
        engine={engine}
        surface={surface}
        onClose={closeSurface}
      />
    </DndContext>
  );
};
