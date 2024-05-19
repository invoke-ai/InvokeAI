import { useGlobalMenuClose, useToken } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useConnection } from 'features/nodes/hooks/useConnection';
import { useCopyPaste } from 'features/nodes/hooks/useCopyPaste';
import { useSyncExecutionState } from 'features/nodes/hooks/useExecutionState';
import { useIsValidConnection } from 'features/nodes/hooks/useIsValidConnection';
import { useWorkflowWatcher } from 'features/nodes/hooks/useWorkflowWatcher';
import {
  $cursorPos,
  $didUpdateEdge,
  $edgePendingUpdate,
  $isAddNodePopoverOpen,
  $lastEdgeUpdateMouseEvent,
  $pendingConnection,
  $viewport,
  edgesChanged,
  nodesChanged,
  redo,
  undo,
} from 'features/nodes/store/nodesSlice';
import { $flow } from 'features/nodes/store/reactFlowInstance';
import { connectionToEdge } from 'features/nodes/store/util/reactFlowUtil';
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
  const nodes = useAppSelector((s) => s.nodes.present.nodes);
  const edges = useAppSelector((s) => s.nodes.present.edges);
  const viewport = useStore($viewport);
  const mayUndo = useAppSelector((s) => s.nodes.past.length > 0);
  const mayRedo = useAppSelector((s) => s.nodes.future.length > 0);
  const shouldSnapToGrid = useAppSelector((s) => s.workflowSettings.shouldSnapToGrid);
  const selectionMode = useAppSelector((s) => s.workflowSettings.selectionMode);
  const { onConnectStart, onConnect, onConnectEnd } = useConnection();
  const flowWrapper = useRef<HTMLDivElement>(null);
  const isValidConnection = useIsValidConnection();
  const cancelConnection = useReactFlowStore(selectCancelConnection);
  const updateNodeInternals = useUpdateNodeInternals();
  const store = useAppStore();
  useWorkflowWatcher();
  useSyncExecutionState();
  const [borderRadius] = useToken('radii', ['base']);

  const flowStyles = useMemo<CSSProperties>(
    () => ({
      borderRadius,
    }),
    [borderRadius]
  );

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      dispatch(nodesChanged(changes));
    },
    [dispatch]
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      dispatch(edgesChanged(changes));
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

  const { copySelection, pasteSelection } = useCopyPaste();

  const onCopyHotkey = useCallback(
    (e: KeyboardEvent) => {
      e.preventDefault();
      copySelection();
    },
    [copySelection]
  );
  useHotkeys(['Ctrl+c', 'Meta+c'], onCopyHotkey);

  const onSelectAllHotkey = useCallback(
    (e: KeyboardEvent) => {
      e.preventDefault();
      const { nodes, edges } = store.getState().nodes.present;
      const nodeChanges: NodeChange[] = [];
      const edgeChanges: EdgeChange[] = [];
      nodes.forEach((n) => {
        nodeChanges.push({ id: n.id, type: 'select', selected: true });
      });
      edges.forEach((e) => {
        edgeChanges.push({ id: e.id, type: 'select', selected: true });
      });
      dispatch(nodesChanged(nodeChanges));
      dispatch(edgesChanged(edgeChanges));
    },
    [dispatch, store]
  );
  useHotkeys(['Ctrl+a', 'Meta+a'], onSelectAllHotkey);

  const onPasteHotkey = useCallback(
    (e: KeyboardEvent) => {
      e.preventDefault();
      pasteSelection();
    },
    [pasteSelection]
  );
  useHotkeys(['Ctrl+v', 'Meta+v'], onPasteHotkey);

  const onPasteWithEdgesToNodesHotkey = useCallback(
    (e: KeyboardEvent) => {
      e.preventDefault();
      pasteSelection(true);
    },
    [pasteSelection]
  );
  useHotkeys(['Ctrl+shift+v', 'Meta+shift+v'], onPasteWithEdgesToNodesHotkey);

  const onUndoHotkey = useCallback(() => {
    if (mayUndo) {
      dispatch(undo());
    }
  }, [dispatch, mayUndo]);
  useHotkeys(['meta+z', 'ctrl+z'], onUndoHotkey);

  const onRedoHotkey = useCallback(() => {
    if (mayRedo) {
      dispatch(redo());
    }
  }, [dispatch, mayRedo]);
  useHotkeys(['meta+shift+z', 'ctrl+shift+z'], onRedoHotkey);

  const onEscapeHotkey = useCallback(() => {
    if (!$edgePendingUpdate.get()) {
      $pendingConnection.set(null);
      $isAddNodePopoverOpen.set(false);
      cancelConnection();
    }
  }, [cancelConnection]);
  useHotkeys('esc', onEscapeHotkey);

  const onDeleteHotkey = useCallback(() => {
    const { nodes, edges } = store.getState().nodes.present;
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
    dispatch(nodesChanged(nodeChanges));
    dispatch(edgesChanged(edgeChanges));
  }, [dispatch, store]);
  useHotkeys(['delete', 'backspace'], onDeleteHotkey);

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
    >
      <Background />
    </ReactFlow>
  );
});

Flow.displayName = 'Flow';
