import { useToken } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { $flow } from 'features/nodes/store/reactFlowInstance';
import { contextMenusClosed } from 'features/ui/store/uiSlice';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  Background,
  OnConnect,
  OnConnectEnd,
  OnConnectStart,
  OnEdgesChange,
  OnEdgesDelete,
  OnInit,
  OnMoveEnd,
  OnNodesChange,
  OnNodesDelete,
  OnSelectionChangeFunc,
  ProOptions,
  ReactFlow,
} from 'reactflow';
import { useIsValidConnection } from '../../hooks/useIsValidConnection';
import {
  connectionEnded,
  connectionMade,
  connectionStarted,
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
} from '../../store/nodesSlice';
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
    dispatch(connectionEnded());
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
    dispatch(contextMenusClosed());
  }, [dispatch]);

  const onInit: OnInit = useCallback((flow) => {
    $flow.set(flow);
    flow.fitView();
  }, []);

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
    dispatch(selectionPasted());
  });

  return (
    <ReactFlow
      id="workflow-editor"
      defaultViewport={viewport}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      onInit={onInit}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onEdgesDelete={onEdgesDelete}
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
