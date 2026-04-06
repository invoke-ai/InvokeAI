import { Menu, MenuButton, MenuItem, MenuList, Portal, useGlobalMenuClose, useToken } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type {
  EdgeChange,
  HandleType,
  NodeChange,
  OnEdgesChange,
  OnInit,
  OnMoveEnd,
  OnNodesChange,
  OnReconnect,
  ProOptions,
  ReactFlowProps,
  ReactFlowState,
} from '@xyflow/react';
import {
  Background,
  ReactFlow,
  SelectionMode,
  useStore as useReactFlowStore,
  useUpdateNodeInternals,
} from '@xyflow/react';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { useConnection } from 'features/nodes/hooks/useConnection';
import { useIsValidConnection } from 'features/nodes/hooks/useIsValidConnection';
import { useNodeCopyPaste } from 'features/nodes/hooks/useNodeCopyPaste';
import {
  $addNodeCmdk,
  $cursorPos,
  $didUpdateEdge,
  $edgePendingUpdate,
  $lastEdgeUpdateMouseEvent,
  $pendingConnection,
  $templates,
  $viewport,
  connectorDeleted,
  connectorInserted,
  edgesChanged,
  nodesChanged,
  redo,
  undo,
} from 'features/nodes/store/nodesSlice';
import { $flow, $needsFit } from 'features/nodes/store/reactFlowInstance';
import {
  selectEdges,
  selectMayRedo,
  selectMayUndo,
  selectNodes,
  selectNodesSlice,
} from 'features/nodes/store/selectors';
import { getConnectorDeletionSpliceConnections } from 'features/nodes/store/util/connectorTopology';
import { connectionToEdge } from 'features/nodes/store/util/reactFlowUtil';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import { selectSelectionMode, selectShouldSnapToGrid } from 'features/nodes/store/workflowSettingsSlice';
import { NO_DRAG_CLASS, NO_PAN_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';
import { buildConnectorNode } from 'features/nodes/util/node/buildConnectorNode';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { CSSProperties, MouseEvent, RefObject } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiPlugsConnectedBold, PiTrashBold } from 'react-icons/pi';

import CustomConnectionLine from './connectionLines/CustomConnectionLine';
import InvocationCollapsedEdge from './edges/InvocationCollapsedEdge';
import InvocationDefaultEdge from './edges/InvocationDefaultEdge';
import ConnectorNode from './nodes/Connector/ConnectorNode';
import CurrentImageNode from './nodes/CurrentImage/CurrentImageNode';
import InvocationNodeWrapper from './nodes/Invocation/InvocationNodeWrapper';
import NotesNode from './nodes/Notes/NotesNode';
import { isWorkflowHotkeyEnabled, shouldIgnoreWorkflowCopyHotkey } from './workflowHotkeys';

const edgeTypes = {
  collapsed: InvocationCollapsedEdge,
  default: InvocationDefaultEdge,
} as const;

const nodeTypes = {
  invocation: InvocationNodeWrapper,
  connector: ConnectorNode,
  current_image: CurrentImageNode,
  notes: NotesNode,
} as const;

// TODO: can we support reactflow? if not, we could style the attribution so it matches the app
const proOptions: ProOptions = { hideAttribution: true };

const snapGrid: [number, number] = [25, 25];

const selectCancelConnection = (state: ReactFlowState) => state.cancelConnection;

type WorkflowContextMenuState =
  | {
      kind: 'pane';
      clientX: number;
      clientY: number;
    }
  | {
      kind: 'connector';
      connectorId: string;
    }
  | null;

export const Flow = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const nodes = useAppSelector(selectNodes);
  const edges = useAppSelector(selectEdges);
  const templates = useStore($templates);
  const viewport = useStore($viewport);
  const shouldSnapToGrid = useAppSelector(selectShouldSnapToGrid);
  const selectionMode = useAppSelector(selectSelectionMode);
  const { onConnectStart, onConnect, onConnectEnd } = useConnection();
  const flowWrapper = useRef<HTMLDivElement>(null);
  const contextMenuStateRef = useRef<WorkflowContextMenuState>(null);
  const isValidConnection = useIsValidConnection();
  const updateNodeInternals = useUpdateNodeInternals();
  const [contextMenuPosition, setContextMenuPosition] = useState<{ x: number; y: number } | null>(null);

  useFocusRegion('workflows', flowWrapper);

  const [borderRadius] = useToken('radii', ['base']);
  const flowStyles = useMemo<CSSProperties>(() => ({ borderRadius }), [borderRadius]);

  const onNodesChange: OnNodesChange<AnyNode> = useCallback(
    (nodeChanges) => {
      dispatch(nodesChanged(nodeChanges));
      const flow = $flow.get();
      if (!flow) {
        return;
      }
      if ($needsFit.get()) {
        $needsFit.set(false);
        flow.fitView();
      }
    },
    [dispatch]
  );

  const onEdgesChange: OnEdgesChange<AnyEdge> = useCallback(
    (changes) => {
      if (changes.length > 0) {
        dispatch(edgesChanged(changes));
      }
    },
    [dispatch]
  );

  const handleMoveEnd: OnMoveEnd = useCallback((e, viewport) => {
    $viewport.set(viewport);
  }, []);

  const { onCloseGlobal } = useGlobalMenuClose();
  const handlePaneClick: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onPaneClick']> = useCallback(
    (event) => {
      if ('button' in event && event.button !== 0) {
        return;
      }
      onCloseGlobal();
      contextMenuStateRef.current = null;
      setContextMenuPosition(null);
    },
    [onCloseGlobal]
  );

  const onInit: OnInit<AnyNode, AnyEdge> = useCallback((flow) => {
    $flow.set(flow);
    flow.fitView();
  }, []);

  const onMouseMove = useCallback((event: MouseEvent<HTMLDivElement>) => {
    if (flowWrapper.current?.getBoundingClientRect()) {
      $cursorPos.set(
        $flow.get()?.screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        }) ?? null
      );
    }
  }, []);

  const addConnectorAtPaneMenuPosition = useCallback(() => {
    const contextMenuState = contextMenuStateRef.current;
    if (contextMenuState?.kind !== 'pane') {
      return;
    }
    const flow = $flow.get();
    if (!flow) {
      return;
    }
    const connector = buildConnectorNode(
      flow.screenToFlowPosition({
        x: contextMenuState.clientX,
        y: contextMenuState.clientY,
      })
    );
    dispatch(nodesChanged([{ type: 'add', item: connector }]));
    contextMenuStateRef.current = null;
    setContextMenuPosition(null);
  }, [dispatch]);

  const deleteConnectorFromContextMenu = useCallback(() => {
    const contextMenuState = contextMenuStateRef.current;
    const connectorSpliceConnections =
      contextMenuState?.kind === 'connector'
        ? getConnectorDeletionSpliceConnections(
            contextMenuState.connectorId,
            nodes,
            edges,
            templates,
            validateConnection
          )
        : null;

    if (contextMenuState?.kind !== 'connector' || !connectorSpliceConnections) {
      return;
    }
    dispatch(
      connectorDeleted({
        connectorId: contextMenuState.connectorId,
        spliceConnections: connectorSpliceConnections,
      })
    );
    contextMenuStateRef.current = null;
    setContextMenuPosition(null);
  }, [dispatch, edges, nodes, templates]);

  const onWorkflowContextMenu = useCallback((event: globalThis.MouseEvent) => {
    if (event.shiftKey) {
      contextMenuStateRef.current = null;
      setContextMenuPosition(null);
      return;
    }

    if (!(event.target instanceof Element)) {
      contextMenuStateRef.current = null;
      setContextMenuPosition(null);
      return;
    }

    event.preventDefault();

    const connectorId = event.target.closest<HTMLElement>('[data-connector-node-id]')?.dataset.connectorNodeId;
    if (connectorId) {
      contextMenuStateRef.current = {
        kind: 'connector',
        connectorId,
      };
      setContextMenuPosition({ x: event.pageX, y: event.pageY });
      return;
    }

    contextMenuStateRef.current = {
      kind: 'pane',
      clientX: event.clientX,
      clientY: event.clientY,
    };
    setContextMenuPosition({ x: event.pageX, y: event.pageY });
  }, []);

  useEffect(() => {
    const onWindowContextMenu = (event: globalThis.MouseEvent) => {
      const wrapper = flowWrapper.current;
      const target = event.target;

      if (!wrapper || !(target instanceof Node) || !wrapper.contains(target)) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();

      onWorkflowContextMenu(event);
    };

    window.addEventListener('contextmenu', onWindowContextMenu, { capture: true });

    return () => {
      window.removeEventListener('contextmenu', onWindowContextMenu, { capture: true });
    };
  }, [onWorkflowContextMenu]);

  const renderContextMenu = useCallback(() => {
    const contextMenuState = contextMenuStateRef.current;

    if (contextMenuState?.kind === 'pane') {
      return (
        <MenuList visibility="visible">
          <MenuItem icon={<PiPlugsConnectedBold />} onClick={addConnectorAtPaneMenuPosition}>
            {t('nodes.addConnector')}
          </MenuItem>
        </MenuList>
      );
    }

    if (contextMenuState?.kind === 'connector') {
      const connectorSpliceConnections = getConnectorDeletionSpliceConnections(
        contextMenuState.connectorId,
        nodes,
        edges,
        templates,
        validateConnection
      );

      return (
        <MenuList visibility="visible">
          <MenuItem
            icon={<PiTrashBold />}
            onClick={deleteConnectorFromContextMenu}
            isDisabled={!connectorSpliceConnections}
            isDestructive
          >
            {t('nodes.deleteConnector')}
          </MenuItem>
        </MenuList>
      );
    }

    return <MenuList visibility="visible" />;
  }, [addConnectorAtPaneMenuPosition, deleteConnectorFromContextMenu, edges, nodes, t, templates]);

  const closeContextMenu = useCallback(() => {
    contextMenuStateRef.current = null;
    setContextMenuPosition(null);
  }, []);

  const onEdgeDoubleClick = useCallback<NonNullable<ReactFlowProps['onEdgeDoubleClick']>>(
    (event, edge) => {
      if (edge.type !== 'default' || edge.hidden) {
        return;
      }
      const flow = $flow.get();
      if (!flow) {
        return;
      }
      const connector = buildConnectorNode(
        flow.screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        })
      );
      dispatch(connectorInserted({ edgeId: edge.id, connector }));
      updateNodeInternals([edge.source, edge.target, connector.id]);
    },
    [dispatch, updateNodeInternals]
  );

  // #region Updatable Edges

  /**
   * Adapted from https://reactflow.dev/docs/examples/edges/updatable-edge/
   * and https://reactflow.dev/docs/examples/edges/delete-edge-on-drop/
   *
   * - Edges can be dragged from one handle to another.
   * - If the user drags the edge away from the node and drops it, delete the edge.
   * - Do not delete the edge if the cursor didn't move (resolves annoying behaviour
   *   where the edge is deleted if you click it accidentally).
   */

  const onReconnectStart = useCallback((event: MouseEvent, edge: AnyEdge, _handleType: HandleType) => {
    $edgePendingUpdate.set(edge);
    $didUpdateEdge.set(false);
    $lastEdgeUpdateMouseEvent.set(event);
  }, []);

  const onReconnect: OnReconnect = useCallback(
    (oldEdge, newConnection) => {
      // This event is fired when an edge update is successful
      $didUpdateEdge.set(true);
      // When an edge update is successful, we need to delete the old edge and create a new one
      const newEdge = connectionToEdge(newConnection);
      dispatch(
        edgesChanged([
          { type: 'remove', id: oldEdge.id },
          { type: 'add', item: newEdge },
        ])
      );
      // Because we shift the position of handles depending on whether a field is connected or not, we must use
      // updateNodeInternals to tell reactflow to recalculate the positions of the handles
      updateNodeInternals([oldEdge.source, oldEdge.target, newEdge.source, newEdge.target]);
    },
    [dispatch, updateNodeInternals]
  );

  const onReconnectEnd: NonNullable<ReactFlowProps['onReconnectEnd']> = useCallback(
    (e, edge, _handleType) => {
      const didUpdateEdge = $didUpdateEdge.get();
      // Fall back to a reasonable default event
      const lastEvent = $lastEdgeUpdateMouseEvent.get() ?? { clientX: 0, clientY: 0 };
      // We have to narrow this event down to MouseEvents - could be TouchEvent
      const didMouseMove =
        !('touches' in e) && Math.hypot(e.clientX - lastEvent.clientX, e.clientY - lastEvent.clientY) > 5;

      // If we got this far and did not successfully update an edge, and the mouse moved away from the handle,
      // the user probably intended to delete the edge
      if (!didUpdateEdge && didMouseMove) {
        dispatch(edgesChanged([{ type: 'remove', id: edge.id }]));
      }

      $edgePendingUpdate.set(null);
      $didUpdateEdge.set(false);
      $pendingConnection.set(null);
      $lastEdgeUpdateMouseEvent.set(null);
    },
    [dispatch]
  );

  // #endregion

  const renderedNodes = useMemo(() => nodes, [nodes]);

  const renderedEdges = useMemo(() => edges, [edges]);
  const contextMenuKey = contextMenuPosition ? `${contextMenuPosition.x}-${contextMenuPosition.y}` : 'closed';

  return (
    <>
      <FlowSurface
        flowWrapper={flowWrapper}
        viewport={viewport}
        renderedNodes={renderedNodes}
        renderedEdges={renderedEdges}
        onInit={onInit}
        onMouseMove={onMouseMove}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onReconnect={onReconnect}
        onReconnectStart={onReconnectStart}
        onReconnectEnd={onReconnectEnd}
        onConnectStart={onConnectStart}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        handleMoveEnd={handleMoveEnd}
        onEdgeDoubleClick={onEdgeDoubleClick}
        isValidConnection={isValidConnection}
        shouldSnapToGrid={shouldSnapToGrid}
        flowStyles={flowStyles}
        handlePaneClick={handlePaneClick}
        selectionMode={selectionMode}
      />
      <Portal>
        <Menu
          key={contextMenuKey}
          isOpen={contextMenuPosition !== null}
          onClose={closeContextMenu}
          placement="auto-start"
          gutter={0}
        >
          <MenuButton
            aria-hidden
            position="absolute"
            left={contextMenuPosition?.x ?? -9999}
            top={contextMenuPosition?.y ?? -9999}
            w={1}
            h={1}
            pointerEvents="none"
            bg="transparent"
          />
          {renderContextMenu()}
        </Menu>
      </Portal>
      <HotkeyIsolator flowWrapper={flowWrapper} />
    </>
  );
});

Flow.displayName = 'Flow';

type FlowSurfaceProps = {
  flowWrapper: { current: HTMLDivElement | null };
  viewport: ReactFlowProps<AnyNode, AnyEdge>['defaultViewport'];
  renderedNodes: AnyNode[];
  renderedEdges: AnyEdge[];
  onInit: OnInit<AnyNode, AnyEdge>;
  onMouseMove: (event: MouseEvent<HTMLDivElement>) => void;
  onNodesChange: OnNodesChange<AnyNode>;
  onEdgesChange: OnEdgesChange<AnyEdge>;
  onReconnect: OnReconnect;
  onReconnectStart: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onReconnectStart']>;
  onReconnectEnd: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onReconnectEnd']>;
  onConnectStart: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onConnectStart']>;
  onConnect: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onConnect']>;
  onConnectEnd: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onConnectEnd']>;
  handleMoveEnd: OnMoveEnd;
  onEdgeDoubleClick: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onEdgeDoubleClick']>;
  isValidConnection: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['isValidConnection']>;
  shouldSnapToGrid: boolean;
  flowStyles: CSSProperties;
  handlePaneClick: NonNullable<ReactFlowProps<AnyNode, AnyEdge>['onPaneClick']>;
  selectionMode: ReturnType<typeof selectSelectionMode>;
};

const FlowSurface = memo((props: FlowSurfaceProps) => {
  const {
    flowWrapper,
    viewport,
    renderedNodes,
    renderedEdges,
    onInit,
    onMouseMove,
    onNodesChange,
    onEdgesChange,
    onReconnect,
    onReconnectStart,
    onReconnectEnd,
    onConnectStart,
    onConnect,
    onConnectEnd,
    handleMoveEnd,
    onEdgeDoubleClick,
    isValidConnection,
    shouldSnapToGrid,
    flowStyles,
    handlePaneClick,
    selectionMode,
  } = props;

  const setFlowWrapperElement = useCallback(
    (el: HTMLDivElement | null) => {
      flowWrapper.current = el;
    },
    [flowWrapper]
  );

  return (
    <div ref={setFlowWrapperElement} style={{ width: '100%', height: '100%' }}>
      <ReactFlow<AnyNode, AnyEdge>
        id="workflow-editor"
        defaultViewport={viewport}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodes={renderedNodes}
        edges={renderedEdges}
        onInit={onInit}
        onMouseMove={onMouseMove}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onReconnect={onReconnect}
        onReconnectStart={onReconnectStart}
        onReconnectEnd={onReconnectEnd}
        onConnectStart={onConnectStart}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        onMoveEnd={handleMoveEnd}
        onEdgeDoubleClick={onEdgeDoubleClick}
        connectionLineComponent={CustomConnectionLine}
        isValidConnection={isValidConnection}
        minZoom={0.1}
        snapToGrid={shouldSnapToGrid}
        snapGrid={snapGrid}
        connectionRadius={30}
        proOptions={proOptions}
        style={flowStyles}
        onPaneClick={handlePaneClick}
        deleteKeyCode={null}
        selectionMode={selectionMode === 'full' ? SelectionMode.Full : SelectionMode.Partial}
        elevateEdgesOnSelect
        nodeDragThreshold={1}
        noDragClassName={NO_DRAG_CLASS}
        noWheelClassName={NO_WHEEL_CLASS}
        noPanClassName={NO_PAN_CLASS}
      >
        <Background gap={snapGrid} offset={snapGrid} />
      </ReactFlow>
    </div>
  );
});

FlowSurface.displayName = 'FlowSurface';

const HotkeyIsolator = memo(({ flowWrapper }: { flowWrapper: RefObject<HTMLDivElement> }) => {
  const mayUndo = useAppSelector(selectMayUndo);
  const mayRedo = useAppSelector(selectMayRedo);

  const cancelConnection = useReactFlowStore(selectCancelConnection);

  const store = useAppStore();
  const isWorkflowsFocused = useIsRegionFocused('workflows');

  const { copySelection, pasteSelection, pasteSelectionWithEdges } = useNodeCopyPaste();

  useRegisteredHotkeys({
    id: 'copySelection',
    category: 'workflows',
    callback: copySelection,
    options: {
      enabled: isWorkflowHotkeyEnabled(isWorkflowsFocused),
      preventDefault: true,
      ignoreEventWhen: () => shouldIgnoreWorkflowCopyHotkey(window.getSelection(), flowWrapper.current),
    },
    dependencies: [copySelection, isWorkflowsFocused],
  });

  const selectAll = useCallback(() => {
    const { nodes, edges } = selectNodesSlice(store.getState());
    const nodeChanges: NodeChange<AnyNode>[] = [];
    const edgeChanges: EdgeChange<AnyEdge>[] = [];
    nodes.forEach(({ id, selected }) => {
      if (!selected) {
        nodeChanges.push({ type: 'select', id, selected: true });
      }
    });
    edges.forEach(({ id, selected }) => {
      if (!selected) {
        edgeChanges.push({ type: 'select', id, selected: true });
      }
    });
    if (nodeChanges.length > 0) {
      store.dispatch(nodesChanged(nodeChanges));
    }
    if (edgeChanges.length > 0) {
      store.dispatch(edgesChanged(edgeChanges));
    }
  }, [store]);
  useRegisteredHotkeys({
    id: 'selectAll',
    category: 'workflows',
    callback: selectAll,
    options: { enabled: isWorkflowHotkeyEnabled(isWorkflowsFocused), preventDefault: true },
    dependencies: [selectAll, isWorkflowsFocused],
  });

  useRegisteredHotkeys({
    id: 'pasteSelection',
    category: 'workflows',
    callback: pasteSelection,
    options: { enabled: isWorkflowHotkeyEnabled(isWorkflowsFocused), preventDefault: true },
    dependencies: [pasteSelection, isWorkflowsFocused],
  });

  useRegisteredHotkeys({
    id: 'pasteSelectionWithEdges',
    category: 'workflows',
    callback: pasteSelectionWithEdges,
    options: { enabled: isWorkflowHotkeyEnabled(isWorkflowsFocused), preventDefault: true },
    dependencies: [pasteSelectionWithEdges, isWorkflowsFocused],
  });

  useRegisteredHotkeys({
    id: 'undo',
    category: 'workflows',
    callback: () => {
      store.dispatch(undo());
    },
    options: { enabled: isWorkflowHotkeyEnabled(isWorkflowsFocused) && mayUndo, preventDefault: true },
    dependencies: [store, mayUndo, isWorkflowsFocused],
  });

  useRegisteredHotkeys({
    id: 'redo',
    category: 'workflows',
    callback: () => {
      store.dispatch(redo());
    },
    options: { enabled: isWorkflowHotkeyEnabled(isWorkflowsFocused) && mayRedo, preventDefault: true },
    dependencies: [store, mayRedo, isWorkflowsFocused],
  });

  const onEscapeHotkey = useCallback(() => {
    if (!$edgePendingUpdate.get()) {
      $pendingConnection.set(null);
      $addNodeCmdk.set(false);
      cancelConnection();
    }
  }, [cancelConnection]);
  useHotkeys('esc', onEscapeHotkey);

  const deleteSelection = useCallback(() => {
    const { nodes, edges } = selectNodesSlice(store.getState());
    const nodeChanges: NodeChange<AnyNode>[] = [];
    const edgeChanges: EdgeChange<AnyEdge>[] = [];
    nodes
      .filter((n) => n.selected)
      .forEach(({ id }) => {
        nodeChanges.push({ type: 'remove', id });
      });
    edges
      .filter((e) => e.selected)
      .forEach(({ id }) => {
        edgeChanges.push({ type: 'remove', id });
      });
    if (nodeChanges.length > 0) {
      store.dispatch(nodesChanged(nodeChanges));
    }
    if (edgeChanges.length > 0) {
      store.dispatch(edgesChanged(edgeChanges));
    }
  }, [store]);
  useRegisteredHotkeys({
    id: 'deleteSelection',
    category: 'workflows',
    callback: deleteSelection,
    options: { preventDefault: true, enabled: isWorkflowHotkeyEnabled(isWorkflowsFocused) },
    dependencies: [deleteSelection, isWorkflowsFocused],
  });

  return null;
});
HotkeyIsolator.displayName = 'HotkeyIsolator';
