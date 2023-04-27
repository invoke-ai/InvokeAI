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
  OnConnectStartParams,
} from 'reactflow';
import { Graph, ImageField } from 'services/api';
import { receivedOpenAPISchema } from 'services/thunks/schema';
import { isFulfilledAnyGraphBuilt } from 'services/thunks/session';
import { InvocationTemplate, InvocationValue } from '../types/types';
import { parseSchema } from '../util/parseSchema';

export type NodesState = {
  nodes: Node<InvocationValue>[];
  edges: Edge[];
  schema: OpenAPIV3.Document | null;
  invocationTemplates: Record<string, InvocationTemplate>;
  connectionStartParams: OnConnectStartParams | null;
  lastGraph: Graph | null;
  shouldShowGraphOverlay: boolean;
};

export const initialNodesState: NodesState = {
  nodes: [],
  edges: [],
  schema: null,
  invocationTemplates: {},
  connectionStartParams: null,
  lastGraph: null,
  shouldShowGraphOverlay: false,
};

const nodesSlice = createSlice({
  name: 'nodes',
  initialState: initialNodesState,
  reducers: {
    nodesChanged: (state, action: PayloadAction<NodeChange[]>) => {
      state.nodes = applyNodeChanges(action.payload, state.nodes);
    },
    nodeAdded: (state, action: PayloadAction<Node<InvocationValue>>) => {
      state.nodes.push(action.payload);
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange[]>) => {
      state.edges = applyEdgeChanges(action.payload, state.edges);
    },
    connectionStarted: (state, action: PayloadAction<OnConnectStartParams>) => {
      state.connectionStartParams = action.payload;
    },
    connectionMade: (state, action: PayloadAction<Connection>) => {
      state.edges = addEdge(action.payload, state.edges);
    },
    connectionEnded: (state) => {
      state.connectionStartParams = null;
    },
    fieldValueChanged: (
      state,
      action: PayloadAction<{
        nodeId: string;
        fieldName: string;
        value:
          | string
          | number
          | boolean
          | Pick<ImageField, 'image_name' | 'image_type'>
          | undefined;
      }>
    ) => {
      const { nodeId, fieldName, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      if (nodeIndex > -1) {
        state.nodes[nodeIndex].data.inputs[fieldName].value = value;
      }
    },
    shouldShowGraphOverlayChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowGraphOverlay = action.payload;
    },
    parsedOpenAPISchema: (state, action: PayloadAction<OpenAPIV3.Document>) => {
      try {
        const parsedSchema = parseSchema(action.payload);
        console.debug('Parsed schema: ', parsedSchema);
        state.invocationTemplates = parsedSchema;
      } catch (err) {
        console.error(err);
      }
    },
  },
  extraReducers(builder) {
    builder.addCase(receivedOpenAPISchema.fulfilled, (state, action) => {
      state.schema = action.payload;
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
  shouldShowGraphOverlayChanged,
  parsedOpenAPISchema,
} = nodesSlice.actions;

export default nodesSlice.reducer;
