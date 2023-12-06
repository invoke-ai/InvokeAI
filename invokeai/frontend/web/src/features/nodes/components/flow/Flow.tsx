import { useToken } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useIsValidConnection } from 'features/nodes/hooks/useIsValidConnection';
import {
  connectionEnded,
  connectionMade,
  connectionStarted,
  edgeAdded,
  edgeChangeStarted,
  edgeDeleted,
  edgesChanged,
  edgesDeleted,
  nodesChanged,
  nodesDeleted,
  selectedAll,
  selectedEdgesChanged,
  selectedNodesChanged,
  selectionCopied,
  selectionPasted,
  viewportChanged,
} from 'features/nodes/store/nodesSlice';
import { $flow } from 'features/nodes/store/reactFlowInstance';
import { bumpGlobalMenuCloseTrigger } from 'features/ui/store/uiSlice';
import { MouseEvent, useCallback, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  Background,
  OnConnect,
  OnConnectEnd,
  OnConnectStart,
  OnEdgeUpdateFunc,
  OnEdgesChange,
  OnEdgesDelete,
  OnInit,
  OnMoveEnd,
  OnNodesChange,
  OnNodesDelete,
  OnSelectionChangeFunc,
  ProOptions,
  ReactFlow,
  ReactFlowProps,
  XYPosition,
} from 'reactflow';
import CustomConnectionLine from './connectionLines/CustomConnectionLine';
import InvocationCollapsedEdge from './edges/InvocationCollapsedEdge';
import InvocationDefaultEdge from './edges/InvocationDefaultEdge';
import CurrentImageNode from './nodes/CurrentImage/CurrentImageNode';
import InvocationNodeWrapper from './nodes/Invocation/InvocationNodeWrapper';
import NotesNode from './nodes/Notes/NotesNode';

const DELETE_KEYS = ['Delete', 'Backspace'];

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

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const { shouldSnapToGrid, selectionMode } = nodes;
    return {
      shouldSnapToGrid,
      selectionMode,
    };
  },
  defaultSelectorOptions
);

export const Flow = () => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector((state) => state.nodes.nodes);
  const edges = useAppSelector((state) => state.nodes.edges);
  const viewport = useAppSelector((state) => state.nodes.viewport);
  const { shouldSnapToGrid, selectionMode } = useAppSelector(selector);
  const flowWrapper = useRef<HTMLDivElement>(null);
  const cursorPosition = useRef<XYPosition>();
  const isValidConnection = useIsValidConnection();

  const [borderRadius] = useToken('radii', ['base']);

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

  const onConnectStart: OnConnectStart = useCallback(
    (event, params) => {
      dispatch(connectionStarted(params));
    },
    [dispatch]
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      dispatch(connectionMade(connection));
    },
    [dispatch]
  );

  const onConnectEnd: OnConnectEnd = useCallback(() => {
    dispatch(connectionEnded({ cursorPosition: cursorPosition.current }));
  }, [dispatch]);

  const onEdgesDelete: OnEdgesDelete = useCallback(
    (edges) => {
      dispatch(edgesDeleted(edges));
    },
    [dispatch]
  );

  const onNodesDelete: OnNodesDelete = useCallback(
    (nodes) => {
      dispatch(nodesDeleted(nodes));
    },
    [dispatch]
  );

  const handleSelectionChange: OnSelectionChangeFunc = useCallback(
    ({ nodes, edges }) => {
      dispatch(selectedNodesChanged(nodes ? nodes.map((n) => n.id) : []));
      dispatch(selectedEdgesChanged(edges ? edges.map((e) => e.id) : []));
    },
    [dispatch]
  );

  const handleMoveEnd: OnMoveEnd = useCallback(
    (e, viewport) => {
      dispatch(viewportChanged(viewport));
    },
    [dispatch]
  );

  const handlePaneClick = useCallback(() => {
    dispatch(bumpGlobalMenuCloseTrigger());
  }, [dispatch]);

  const onInit: OnInit = useCallback((flow) => {
    $flow.set(flow);
    flow.fitView();
  }, []);

  const onMouseMove = useCallback((event: MouseEvent<HTMLDivElement>) => {
    const bounds = flowWrapper.current?.getBoundingClientRect();
    if (bounds) {
      const pos = $flow.get()?.project({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });
      cursorPosition.current = pos;
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

  // We have a ref for cursor position, but it is the *projected* cursor position.
  // Easiest to just keep track of the last mouse event for this particular feature
  const edgeUpdateMouseEvent = useRef<MouseEvent>();

  const onEdgeUpdateStart: NonNullable<ReactFlowProps['onEdgeUpdateStart']> =
    useCallback(
      (e, edge, _handleType) => {
        // update mouse event
        edgeUpdateMouseEvent.current = e;
        // always delete the edge when starting an updated
        dispatch(edgeDeleted(edge.id));
        dispatch(edgeChangeStarted());
      },
      [dispatch]
    );

  const onEdgeUpdate: OnEdgeUpdateFunc = useCallback(
    (_oldEdge, newConnection) => {
      // instead of updating the edge (we deleted it earlier), we instead create
      // a new one.
      dispatch(connectionMade(newConnection));
    },
    [dispatch]
  );

  const onEdgeUpdateEnd: NonNullable<ReactFlowProps['onEdgeUpdateEnd']> =
    useCallback(
      (e, edge, _handleType) => {
        // Handle the case where user begins a drag but didn't move the cursor -
        // bc we deleted the edge, we need to add it back
        if (
          // ignore touch events
          !('touches' in e) &&
          edgeUpdateMouseEvent.current?.clientX === e.clientX &&
          edgeUpdateMouseEvent.current?.clientY === e.clientY
        ) {
          dispatch(edgeAdded(edge));
        }
        // reset mouse event
        edgeUpdateMouseEvent.current = undefined;
      },
      [dispatch]
    );

  // #endregion

  useHotkeys(['Ctrl+c', 'Meta+c'], (e) => {
    e.preventDefault();
    dispatch(selectionCopied());
  });

  useHotkeys(['Ctrl+a', 'Meta+a'], (e) => {
    e.preventDefault();
    dispatch(selectedAll());
  });

  useHotkeys(['Ctrl+v', 'Meta+v'], (e) => {
    e.preventDefault();
    dispatch(selectionPasted({ cursorPosition: cursorPosition.current }));
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
      onEdgesDelete={onEdgesDelete}
      onEdgeUpdate={onEdgeUpdate}
      onEdgeUpdateStart={onEdgeUpdateStart}
      onEdgeUpdateEnd={onEdgeUpdateEnd}
      onNodesDelete={onNodesDelete}
      onConnectStart={onConnectStart}
      onConnect={onConnect}
      onConnectEnd={onConnectEnd}
      onMoveEnd={handleMoveEnd}
      connectionLineComponent={CustomConnectionLine}
      onSelectionChange={handleSelectionChange}
      isValidConnection={isValidConnection}
      minZoom={0.1}
      snapToGrid={shouldSnapToGrid}
      snapGrid={[25, 25]}
      connectionRadius={30}
      proOptions={proOptions}
      style={{ borderRadius }}
      onPaneClick={handlePaneClick}
      deleteKeyCode={DELETE_KEYS}
      selectionMode={selectionMode}
    >
      <Background />
    </ReactFlow>
  );
};
