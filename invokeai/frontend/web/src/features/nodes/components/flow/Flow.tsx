import { useGlobalMenuClose, useToken } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { useConnection } from 'features/nodes/hooks/useConnection';
import { useCopyPaste } from 'features/nodes/hooks/useCopyPaste';
import { useSyncExecutionState } from 'features/nodes/hooks/useExecutionState';
import { useIsValidConnection } from 'features/nodes/hooks/useIsValidConnection';
import { useWorkflowWatcher } from 'features/nodes/hooks/useWorkflowWatcher';
import {
  $addNodeCmdk,
  $cursorPos,
  $didUpdateEdge,
  $edgePendingUpdate,
  $lastEdgeUpdateMouseEvent,
  $pendingConnection,
  $viewport,
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
import { connectionToEdge } from 'features/nodes/store/util/reactFlowUtil';
import { selectSelectionMode, selectShouldSnapToGrid } from 'features/nodes/store/workflowSettingsSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { CSSProperties, MouseEvent } from 'react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import type {
  EdgeChange,
  NodeChange,
  OnEdgesChange,
  OnEdgeUpdateFunc,
  OnInit,
  OnMoveEnd,
  OnNodesChange,
  ProOptions,
  ReactFlowProps,
  ReactFlowState,
} from 'reactflow';
import { Background, ReactFlow, useStore as useReactFlowStore, useUpdateNodeInternals } from 'reactflow';

import CustomConnectionLine from './connectionLines/CustomConnectionLine';
import InvocationCollapsedEdge from './edges/InvocationCollapsedEdge';
import InvocationDefaultEdge from './edges/InvocationDefaultEdge';
import CurrentImageNode from './nodes/CurrentImage/CurrentImageNode';
import InvocationNodeWrapper from './nodes/Invocation/InvocationNodeWrapper';
import NotesNode from './nodes/Notes/NotesNode';

const edgeTypes = {
  collapsed: InvocationCollapsedEdge,
  default: InvocationDefaultEdge,
};

const nodeTypes = {
  invocation: InvocationNodeWrapper,
  current_image: CurrentImageNode,
  notes: NotesNode,
};

// TODO: can we support reactflow? if not, we could style the attribution so it matches the app
const proOptions: ProOptions = { hideAttribution: true };

const snapGrid: [number, number] = [25, 25];

const selectCancelConnection = (state: ReactFlowState) => state.cancelConnection;

export const Flow = memo(() => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector(selectNodes);
  const edges = useAppSelector(selectEdges);
  const viewport = useStore($viewport);
  const needsFit = useStore($needsFit);
  const mayUndo = useAppSelector(selectMayUndo);
  const mayRedo = useAppSelector(selectMayRedo);
  const shouldSnapToGrid = useAppSelector(selectShouldSnapToGrid);
  const selectionMode = useAppSelector(selectSelectionMode);
  const { onConnectStart, onConnect, onConnectEnd } = useConnection();
  const flowWrapper = useRef<HTMLDivElement>(null);
  const isValidConnection = useIsValidConnection();
  const cancelConnection = useReactFlowStore(selectCancelConnection);
  const updateNodeInternals = useUpdateNodeInternals();
  const store = useAppStore();
  const isWorkflowsFocused = useIsRegionFocused('workflows');
  useFocusRegion('workflows', flowWrapper);

  useWorkflowWatcher();
  useSyncExecutionState();
  const [borderRadius] = useToken('radii', ['base']);
  const flowStyles = useMemo<CSSProperties>(() => ({ borderRadius }), [borderRadius]);

  const onNodesChange: OnNodesChange = useCallback(
    (nodeChanges) => {
      dispatch(nodesChanged(nodeChanges));
      const flow = $flow.get();
      if (!flow) {
        return;
      }
      if (needsFit) {
        $needsFit.set(false);
        flow.fitView();
      }
    },
    [dispatch, needsFit]
  );

  const onEdgesChange: OnEdgesChange = useCallback(
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
  const handlePaneClick = useCallback(() => {
    onCloseGlobal();
  }, [onCloseGlobal]);

  const onInit: OnInit = useCallback((flow) => {
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

  const onEdgeUpdateStart: NonNullable<ReactFlowProps['onEdgeUpdateStart']> = useCallback((e, edge, _handleType) => {
    $edgePendingUpdate.set(edge);
    $didUpdateEdge.set(false);
    $lastEdgeUpdateMouseEvent.set(e);
  }, []);

  const onEdgeUpdate: OnEdgeUpdateFunc = useCallback(
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

  const onEdgeUpdateEnd: NonNullable<ReactFlowProps['onEdgeUpdateEnd']> = useCallback(
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

  const { copySelection, pasteSelection, pasteSelectionWithEdges } = useCopyPaste();

  useRegisteredHotkeys({
    id: 'copySelection',
    category: 'workflows',
    callback: copySelection,
    options: { preventDefault: true },
    dependencies: [copySelection],
  });

  const selectAll = useCallback(() => {
    const { nodes, edges } = selectNodesSlice(store.getState());
    const nodeChanges: NodeChange[] = [];
    const edgeChanges: EdgeChange[] = [];
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
      dispatch(nodesChanged(nodeChanges));
    }
    if (edgeChanges.length > 0) {
      dispatch(edgesChanged(edgeChanges));
    }
  }, [dispatch, store]);
  useRegisteredHotkeys({
    id: 'selectAll',
    category: 'workflows',
    callback: selectAll,
    options: { enabled: isWorkflowsFocused, preventDefault: true },
    dependencies: [selectAll, isWorkflowsFocused],
  });

  useRegisteredHotkeys({
    id: 'pasteSelection',
    category: 'workflows',
    callback: pasteSelection,
    options: { preventDefault: true },
    dependencies: [pasteSelection],
  });

  useRegisteredHotkeys({
    id: 'pasteSelectionWithEdges',
    category: 'workflows',
    callback: pasteSelectionWithEdges,
    options: { preventDefault: true },
    dependencies: [pasteSelectionWithEdges],
  });

  useRegisteredHotkeys({
    id: 'undo',
    category: 'workflows',
    callback: () => {
      dispatch(undo());
    },
    options: { enabled: mayUndo, preventDefault: true },
    dependencies: [mayUndo],
  });

  useRegisteredHotkeys({
    id: 'redo',
    category: 'workflows',
    callback: () => {
      dispatch(redo());
    },
    options: { enabled: mayRedo, preventDefault: true },
    dependencies: [mayRedo],
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
    const nodeChanges: NodeChange[] = [];
    const edgeChanges: EdgeChange[] = [];
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
      dispatch(nodesChanged(nodeChanges));
    }
    if (edgeChanges.length > 0) {
      dispatch(edgesChanged(edgeChanges));
    }
  }, [dispatch, store]);
  useRegisteredHotkeys({
    id: 'deleteSelection',
    category: 'workflows',
    callback: deleteSelection,
    options: { preventDefault: true, enabled: isWorkflowsFocused },
    dependencies: [deleteSelection, isWorkflowsFocused],
  });

  return (
    <ReactFlow
      id="workflow-editor"
      ref={flowWrapper}
      defaultViewport={viewport}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      onInit={onInit}
      onMouseMove={onMouseMove}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onEdgeUpdate={onEdgeUpdate}
      onEdgeUpdateStart={onEdgeUpdateStart}
      onEdgeUpdateEnd={onEdgeUpdateEnd}
      onConnectStart={onConnectStart}
      onConnect={onConnect}
      onConnectEnd={onConnectEnd}
      onMoveEnd={handleMoveEnd}
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
      selectionMode={selectionMode}
      elevateEdgesOnSelect
      nodeDragThreshold={1}
    >
      <Background />
    </ReactFlow>
  );
});

Flow.displayName = 'Flow';
