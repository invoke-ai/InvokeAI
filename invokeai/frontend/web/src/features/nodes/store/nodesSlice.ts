import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { OpenAPIV3 } from 'openapi-types';
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Connection,
  Edge,
  EdgeChange,
  Node,
  NodeChange,
  NodeTypes,
} from 'reactflow';
import { receivedOpenAPISchema } from 'services/thunks/schema';
import { Invocation } from '../types';
import { parseSchema } from '../util/parseSchema';

export type NodesState = {
  nodes: Node[];
  edges: Edge[];
  schema: OpenAPIV3.Document | null;
  invocations: Record<string, Invocation>;
};

export const initialNodesState: NodesState = {
  nodes: [],
  edges: [],
  schema: null,
  invocations: {},
};

const nodesSlice = createSlice({
  name: 'results',
  initialState: initialNodesState,
  reducers: {
    nodesChanged: (state, action: PayloadAction<NodeChange[]>) => {
      state.nodes = applyNodeChanges(action.payload, state.nodes);
    },
    nodeAdded: (
      state,
      action: PayloadAction<{ id: string; invocation: Invocation }>
    ) => {
      const { id, invocation } = action.payload;

      const node: Node = {
        id,
        type: 'invocation',
        position: { x: 0, y: 0 },
        data: invocation,
      };

      state.nodes.push(node);
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange[]>) => {
      state.edges = applyEdgeChanges(action.payload, state.edges);
    },
    connectionMade: (state, action: PayloadAction<Connection>) => {
      state.edges = addEdge(action.payload, state.edges);
    },
  },
  extraReducers(builder) {
    builder.addCase(receivedOpenAPISchema.fulfilled, (state, action) => {
      console.log('schema received');
      state.schema = action.payload;
      state.invocations = parseSchema(action.payload);
    });
  },
});

export const { nodesChanged, edgesChanged, nodeAdded } = nodesSlice.actions;

export default nodesSlice.reducer;
