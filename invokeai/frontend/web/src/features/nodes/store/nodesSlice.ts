import { createSlice, isAnyOf, PayloadAction } from '@reduxjs/toolkit';
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
  OnConnectStartParams,
} from 'reactflow';
import { Graph } from 'services/api';
import { receivedOpenAPISchema } from 'services/thunks/schema';
import {
  isFulfilledAnyGraphBuilt,
  linearGraphBuilt,
  nodesGraphBuilt,
} from 'services/thunks/session';
import { Invocation } from '../types';
import { buildNodesGraph } from '../util/buildNodesGraph';
import { parseSchema } from '../util/parseSchema';

export type NodesState = {
  nodes: Node<Invocation>[];
  edges: Edge[];
  schema: OpenAPIV3.Document | null;
  invocations: Record<string, Invocation>;
  pendingConnection: OnConnectStartParams | null;
  lastGraph: Graph | null;
};

export const initialNodesState: NodesState = {
  nodes: [],
  edges: [],
  schema: null,
  invocations: {},
  pendingConnection: null,
  lastGraph: null,
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
    connectionStarted: (state, action: PayloadAction<OnConnectStartParams>) => {
      state.pendingConnection = action.payload;
    },
    connectionMade: (state, action: PayloadAction<Connection>) => {
      state.edges = addEdge(action.payload, state.edges);
    },
    connectionEnded: (state) => {
      state.pendingConnection = null;
    },
    fieldValueChanged: (
      state,
      action: PayloadAction<{
        nodeId: string;
        fieldId: string;
        value: string | number | boolean | undefined;
      }>
    ) => {
      const { nodeId, fieldId, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      if (nodeIndex > -1) {
        state.nodes[nodeIndex].data.inputs[fieldId].value = value;
      }
    },
  },
  extraReducers(builder) {
    builder.addCase(receivedOpenAPISchema.fulfilled, (state, action) => {
      state.schema = action.payload;
      state.invocations = parseSchema(action.payload);
    });

    builder.addMatcher(isFulfilledAnyGraphBuilt, (state, action) => {
      state.lastGraph = action.payload;
    });
  },
});

export const {
  nodesChanged,
  edgesChanged,
  nodeAdded,
  fieldValueChanged,
  connectionMade,
  connectionStarted,
  connectionEnded,
} = nodesSlice.actions;

export default nodesSlice.reducer;
