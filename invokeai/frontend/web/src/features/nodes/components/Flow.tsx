import {
  Background,
  Controls,
  MiniMap,
  OnConnect,
  OnEdgesChange,
  OnNodesChange,
  ReactFlow,
  ConnectionLineType,
  OnConnectStart,
  OnConnectEnd,
  Panel,
} from 'reactflow';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { RootState } from 'app/store';
import {
  connectionEnded,
  connectionMade,
  connectionStarted,
  edgesChanged,
  nodesChanged,
} from '../store/nodesSlice';
import { useCallback, useState } from 'react';
import { InvocationComponent } from './InvocationComponent';
import { AddNodeMenu } from './AddNodeMenu';
import { FieldTypeLegend } from './FieldTypeLegend';
import { Button } from '@chakra-ui/react';
import { nodesGraphBuilt } from 'services/thunks/session';
import { IAIIconButton } from 'exports';
import { InfoIcon } from '@chakra-ui/icons';
import { ViewportControls } from './ViewportControls';
import NodeGraphOverlay from './NodeGraphOverlay';
import TopLeftPanel from './panels/TopLeftPanel';
import TopRightPanel from './panels/TopRightPanel';
import TopCenterPanel from './panels/TopCenterPanel';
import BottomLeftPanel from './panels/BottomLeftPanel.tsx';

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

  const onConnectEnd: OnConnectEnd = useCallback(
    (event) => {
      dispatch(connectionEnded());
    },
    [dispatch]
  );

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
      <MiniMap nodeStrokeWidth={3} zoomable pannable />
    </ReactFlow>
  );
};
