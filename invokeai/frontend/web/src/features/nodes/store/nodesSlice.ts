import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Connection,
  Edge,
  EdgeChange,
  Node,
  NodeChange,
} from 'reactflow';

export type NodesState = {
  nodes: Node[];
  edges: Edge[];
};

export const initialNodesState: NodesState = {
  nodes: [],
  edges: [],
};

const nodesSlice = createSlice({
  name: 'results',
  initialState: initialNodesState,
  reducers: {
    nodesChanged: (state, action: PayloadAction<NodeChange[]>) => {
      state.nodes = applyNodeChanges(action.payload, state.nodes);
    },
    nodeAdded: (state, action: PayloadAction<Node>) => {
      state.nodes.push(action.payload);
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange[]>) => {
      state.edges = applyEdgeChanges(action.payload, state.edges);
    },
    connectionMade: (state, action: PayloadAction<Connection>) => {
      state.edges = addEdge(action.payload, state.edges);
    },
  },
});

export const { nodesChanged, edgesChanged, nodeAdded } = nodesSlice.actions;

export default nodesSlice.reducer;
