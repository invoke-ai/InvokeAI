import { useToken } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { contextMenusClosed } from 'features/ui/store/uiSlice';
import { useCallback } from 'react';
import {
  Background,
  OnConnect,
  OnConnectEnd,
  OnConnectStart,
  OnEdgesChange,
  OnEdgesDelete,
  OnMoveEnd,
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
  viewportChanged,
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
  const viewport = useAppSelector((state) => state.nodes.viewport);
  const shouldSnapToGrid = useAppSelector(
    (state) => state.nodes.shouldSnapToGrid
  );

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

  return (
    <ReactFlow
      defaultViewport={viewport}
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
