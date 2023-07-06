import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { cloneDeep, uniqBy } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { RgbaColor } from 'react-colorful';
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
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { ImageField } from 'services/api/types';
import { InvocationTemplate, InvocationValue } from '../types/types';

export type NodesState = {
  nodes: Node<InvocationValue>[];
  edges: Edge[];
  schema: OpenAPIV3.Document | null;
  invocationTemplates: Record<string, InvocationTemplate>;
  connectionStartParams: OnConnectStartParams | null;
  shouldShowGraphOverlay: boolean;
};

export const initialNodesState: NodesState = {
  nodes: [],
  edges: [],
  schema: null,
  invocationTemplates: {},
  connectionStartParams: null,
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
          | ImageField
          | RgbaColor
          | undefined
          | ImageField[];
      }>
    ) => {
      const { nodeId, fieldName, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      if (nodeIndex > -1) {
        state.nodes[nodeIndex].data.inputs[fieldName].value = value;
      }
    },
    imageCollectionFieldValueChanged: (
      state,
      action: PayloadAction<{
        nodeId: string;
        fieldName: string;
        value: ImageField[];
      }>
    ) => {
      const { nodeId, fieldName, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      if (nodeIndex === -1) {
        return;
      }

      const currentValue = cloneDeep(
        state.nodes[nodeIndex].data.inputs[fieldName].value
      );

      if (!currentValue) {
        state.nodes[nodeIndex].data.inputs[fieldName].value = value;
        return;
      }

      state.nodes[nodeIndex].data.inputs[fieldName].value = uniqBy(
        (currentValue as ImageField[]).concat(value),
        'image_name'
      );
    },
    shouldShowGraphOverlayChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowGraphOverlay = action.payload;
    },
    nodeTemplatesBuilt: (
      state,
      action: PayloadAction<Record<string, InvocationTemplate>>
    ) => {
      state.invocationTemplates = action.payload;
    },
    nodeEditorReset: () => {
      return { ...initialNodesState };
    },
  },
  extraReducers: (builder) => {
    builder.addCase(receivedOpenAPISchema.fulfilled, (state, action) => {
      state.schema = action.payload;
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
  nodeTemplatesBuilt,
  nodeEditorReset,
  imageCollectionFieldValueChanged,
} = nodesSlice.actions;

export default nodesSlice.reducer;

export const nodesSelector = (state: RootState) => state.nodes;
