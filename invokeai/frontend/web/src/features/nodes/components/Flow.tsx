import {
  Background,
  Controls,
  MiniMap,
  OnEdgesChange,
  OnNodesChange,
  ReactFlow,
} from 'reactflow';
import { NODE_TYPES } from '../constants';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { RootState } from 'app/store';
import { edgesChanged, nodesChanged } from '../store/nodesSlice';
import { useCallback } from 'react';

export const Flow = () => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector((state: RootState) => state.nodes.nodes);
  const edges = useAppSelector((state: RootState) => state.nodes.edges);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => dispatch(nodesChanged(changes)),
    [dispatch]
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => dispatch(edgesChanged(changes)),
    [dispatch]
  );

  return (
    <ReactFlow
      nodeTypes={NODE_TYPES}
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
    >
      <Background />
      <Controls />
      <MiniMap nodeStrokeWidth={3} zoomable pannable />
    </ReactFlow>
  );
};
