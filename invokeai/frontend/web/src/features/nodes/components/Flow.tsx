import {
  Background,
  OnConnect,
  OnEdgesChange,
  OnNodesChange,
  ReactFlow,
  OnConnectStart,
  OnConnectEnd,
} from 'reactflow';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RootState } from 'app/store/store';
import {
  connectionEnded,
  connectionMade,
  connectionStarted,
  edgesChanged,
  nodesChanged,
} from '../store/nodesSlice';
import { useCallback } from 'react';
import { InvocationComponent } from './InvocationComponent';
import TopLeftPanel from './panels/TopLeftPanel';
import TopRightPanel from './panels/TopRightPanel';
import TopCenterPanel from './panels/TopCenterPanel';
import BottomLeftPanel from './panels/BottomLeftPanel.tsx';
import MinimapPanel from './panels/MinimapPanel';

const nodeTypes = { invocation: InvocationComponent };

export const Flow = () => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector((state: RootState) => state.nodes.nodes);
  const edges = useAppSelector((state: RootState) => state.nodes.edges);

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

  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnectStart={onConnectStart}
      onConnect={onConnect}
      onConnectEnd={onConnectEnd}
      defaultEdgeOptions={{
        style: { strokeWidth: 2 },
      }}
    >
      <TopLeftPanel />
      <TopCenterPanel />
      <TopRightPanel />
      <BottomLeftPanel />
      <Background />
      <MinimapPanel />
    </ReactFlow>
  );
};
