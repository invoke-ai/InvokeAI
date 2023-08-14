import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import {
  Background,
  OnConnect,
  OnConnectEnd,
  OnConnectStart,
  OnEdgesChange,
  OnEdgesDelete,
  OnInit,
  OnMove,
  OnNodesChange,
  OnNodesDelete,
  OnSelectionChangeFunc,
  ProOptions,
  ReactFlow,
} from 'reactflow';
import { useIsValidConnection } from '../hooks/useIsValidConnection';
import {
  connectionEnded,
  connectionMade,
  connectionStarted,
  edgesChanged,
  edgesDeleted,
  nodesChanged,
  nodesDeleted,
  selectedEdgesChanged,
  selectedNodesChanged,
  zoomChanged,
} from '../store/nodesSlice';
import { CustomConnectionLine } from './CustomConnectionLine';
import { edgeTypes } from './CustomEdges';
import { nodeTypes } from './CustomNodes';
import BottomLeftPanel from './editorPanels/BottomLeftPanel';
import MinimapPanel from './editorPanels/MinimapPanel';
import TopCenterPanel from './editorPanels/TopCenterPanel';
import TopLeftPanel from './editorPanels/TopLeftPanel';
import TopRightPanel from './editorPanels/TopRightPanel';

// TODO: can we support reactflow? if not, we could style the attribution so it matches the app
const proOptions: ProOptions = { hideAttribution: true };

export const Flow = () => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector((state) => state.nodes.nodes);
  const edges = useAppSelector((state) => state.nodes.edges);
  const shouldSnapToGrid = useAppSelector(
    (state) => state.nodes.shouldSnapToGrid
  );

  const isValidConnection = useIsValidConnection();

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

  const onInit: OnInit = useCallback((v) => {
    v.fitView();
  }, []);

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

  const handleMove: OnMove = useCallback(
    (e, viewport) => {
      const { zoom } = viewport;
      dispatch(zoomChanged(zoom));
    },
    [dispatch]
  );

  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onEdgesDelete={onEdgesDelete}
      onNodesDelete={onNodesDelete}
      onConnectStart={onConnectStart}
      onConnect={onConnect}
      onConnectEnd={onConnectEnd}
      onMove={handleMove}
      connectionLineComponent={CustomConnectionLine}
      onSelectionChange={handleSelectionChange}
      onInit={onInit}
      isValidConnection={isValidConnection}
      minZoom={0.2}
      snapToGrid={shouldSnapToGrid}
      snapGrid={[25, 25]}
      connectionRadius={30}
      proOptions={proOptions}
    >
      <TopLeftPanel />
      <TopCenterPanel />
      <TopRightPanel />
      <BottomLeftPanel />
      <MinimapPanel />
      <Background />
    </ReactFlow>
  );
};
