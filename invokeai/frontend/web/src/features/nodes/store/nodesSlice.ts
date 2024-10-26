import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig } from 'app/store/store';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { workflowLoaded } from 'features/nodes/store/actions';
import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type {
  BoardFieldValue,
  BooleanFieldValue,
  CLIPEmbedModelFieldValue,
  ColorFieldValue,
  ControlNetModelFieldValue,
  EnumFieldValue,
  FieldValue,
  FloatFieldValue,
  FluxVAEModelFieldValue,
  ImageFieldValue,
  IntegerFieldValue,
  IPAdapterModelFieldValue,
  LoRAModelFieldValue,
  MainModelFieldValue,
  ModelIdentifierFieldValue,
  SchedulerFieldValue,
  SDXLRefinerModelFieldValue,
  SpandrelImageToImageModelFieldValue,
  StatefulFieldValue,
  StringFieldValue,
  T2IAdapterModelFieldValue,
  T5EncoderModelFieldValue,
  VAEModelFieldValue,
} from 'features/nodes/types/field';
import {
  zBoardFieldValue,
  zBooleanFieldValue,
  zCLIPEmbedModelFieldValue,
  zColorFieldValue,
  zControlNetModelFieldValue,
  zEnumFieldValue,
  zFloatFieldValue,
  zFluxVAEModelFieldValue,
  zImageFieldValue,
  zIntegerFieldValue,
  zIPAdapterModelFieldValue,
  zLoRAModelFieldValue,
  zMainModelFieldValue,
  zModelIdentifierFieldValue,
  zSchedulerFieldValue,
  zSDXLRefinerModelFieldValue,
  zSpandrelImageToImageModelFieldValue,
  zStatefulFieldValue,
  zStringFieldValue,
  zT2IAdapterModelFieldValue,
  zT5EncoderModelFieldValue,
  zVAEModelFieldValue,
} from 'features/nodes/types/field';
import type { AnyNode, InvocationNodeEdge } from 'features/nodes/types/invocation';
import { isInvocationNode, isNotesNode } from 'features/nodes/types/invocation';
import { atom, computed } from 'nanostores';
import type { MouseEvent } from 'react';
import type { Edge, EdgeChange, NodeChange, Viewport, XYPosition } from 'reactflow';
import { applyEdgeChanges, applyNodeChanges, getConnectedEdges, getIncomers, getOutgoers } from 'reactflow';
import type { UndoableOptions } from 'redux-undo';
import type { z } from 'zod';

import type { NodesState, PendingConnection, Templates } from './types';

const initialNodesState: NodesState = {
  _version: 1,
  nodes: [],
  edges: [],
};

type FieldValueAction<T extends FieldValue> = PayloadAction<{
  nodeId: string;
  fieldName: string;
  value: T;
}>;

const fieldValueReducer = <T extends FieldValue>(
  state: NodesState,
  action: FieldValueAction<T>,
  schema: z.ZodTypeAny
) => {
  const { nodeId, fieldName, value } = action.payload;
  const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
  const node = state.nodes?.[nodeIndex];
  if (!isInvocationNode(node)) {
    return;
  }
  const input = node.data?.inputs[fieldName];
  const result = schema.safeParse(value);
  if (!input || nodeIndex < 0 || !result.success) {
    return;
  }
  input.value = result.data;
};

export const nodesSlice = createSlice({
  name: 'nodes',
  initialState: initialNodesState,
  reducers: {
    nodesChanged: (state, action: PayloadAction<NodeChange[]>) => {
      state.nodes = applyNodeChanges(action.payload, state.nodes);
      // Remove edges that are no longer valid, due to a removed or otherwise changed node
      const edgeChanges: EdgeChange[] = [];
      state.edges.forEach((e) => {
        const sourceExists = state.nodes.some((n) => n.id === e.source);
        const targetExists = state.nodes.some((n) => n.id === e.target);
        if (!(sourceExists && targetExists)) {
          edgeChanges.push({ type: 'remove', id: e.id });
        }
      });
      state.edges = applyEdgeChanges(edgeChanges, state.edges);
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange[]>) => {
      const changes: EdgeChange[] = [];
      // We may need to massage the edge changes or otherwise handle them
      action.payload.forEach((change) => {
        if (change.type === 'remove' || change.type === 'select') {
          const edge = state.edges.find((e) => e.id === change.id);
          // If we deleted or selected a collapsed edge, we need to find its "hidden" edges and do the same to them
          if (edge && edge.type === 'collapsed') {
            const hiddenEdges = state.edges.filter((e) => e.source === edge.source && e.target === edge.target);
            if (change.type === 'remove') {
              hiddenEdges.forEach(({ id }) => {
                changes.push({ type: 'remove', id });
              });
            }
            if (change.type === 'select') {
              hiddenEdges.forEach(({ id }) => {
                changes.push({ type: 'select', id, selected: change.selected });
              });
            }
          }
        }
        if (change.type === 'add') {
          if (!change.item.type) {
            // We must add the edge type!
            change.item.type = 'default';
          }
        }
        changes.push(change);
      });
      state.edges = applyEdgeChanges(changes, state.edges);
    },
    fieldLabelChanged: (
      state,
      action: PayloadAction<{
        nodeId: string;
        fieldName: string;
        label: string;
      }>
    ) => {
      const { nodeId, fieldName, label } = action.payload;
      const node = state.nodes.find((n) => n.id === nodeId);
      if (!isInvocationNode(node)) {
        return;
      }
      const field = node.data.inputs[fieldName];
      if (!field) {
        return;
      }
      field.label = label;
    },
    nodeUseCacheChanged: (state, action: PayloadAction<{ nodeId: string; useCache: boolean }>) => {
      const { nodeId, useCache } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.useCache = useCache;
    },
    nodeIsIntermediateChanged: (state, action: PayloadAction<{ nodeId: string; isIntermediate: boolean }>) => {
      const { nodeId, isIntermediate } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.isIntermediate = isIntermediate;
    },
    nodeIsOpenChanged: (state, action: PayloadAction<{ nodeId: string; isOpen: boolean }>) => {
      const { nodeId, isOpen } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node) && !isNotesNode(node)) {
        return;
      }

      node.data.isOpen = isOpen;

      if (!isInvocationNode(node)) {
        return;
      }

      // edges between two closed nodes should not be visible:
      // - if the node was just opened, we need to make all its edges visible
      // - if the edge was just closed, we need to check all its edges and hide them if both nodes are closed

      const connectedEdges = getConnectedEdges([node], state.edges);

      if (isOpen) {
        // reset hidden status of all edges
        connectedEdges.forEach((edge) => {
          delete edge.hidden;
        });
        // delete dummy edges
        connectedEdges.forEach((edge) => {
          if (edge.type === 'collapsed') {
            state.edges = state.edges.filter((e) => e.id !== edge.id);
          }
        });
      } else {
        const closedIncomers = getIncomers(node, state.nodes, state.edges).filter(
          (node) => isInvocationNode(node) && node.data.isOpen === false
        );

        const closedOutgoers = getOutgoers(node, state.nodes, state.edges).filter(
          (node) => isInvocationNode(node) && node.data.isOpen === false
        );

        const collapsedEdgesToCreate: Edge<{ count: number }>[] = [];

        // hide all edges
        connectedEdges.forEach((edge) => {
          if (edge.target === nodeId && closedIncomers.find((node) => node.id === edge.source)) {
            edge.hidden = true;
            const collapsedEdge = collapsedEdgesToCreate.find(
              (e) => e.source === edge.source && e.target === edge.target
            );
            if (collapsedEdge) {
              collapsedEdge.data = {
                count: (collapsedEdge.data?.count ?? 0) + 1,
              };
            } else {
              collapsedEdgesToCreate.push({
                id: `${edge.source}-${edge.target}-collapsed`,
                source: edge.source,
                target: edge.target,
                type: 'collapsed',
                data: { count: 1 },
                updatable: false,
                selected: edge.selected,
              });
            }
          }
          if (edge.source === nodeId && closedOutgoers.find((node) => node.id === edge.target)) {
            const collapsedEdge = collapsedEdgesToCreate.find(
              (e) => e.source === edge.source && e.target === edge.target
            );
            edge.hidden = true;
            if (collapsedEdge) {
              collapsedEdge.data = {
                count: (collapsedEdge.data?.count ?? 0) + 1,
              };
            } else {
              collapsedEdgesToCreate.push({
                id: `${edge.source}-${edge.target}-collapsed`,
                source: edge.source,
                target: edge.target,
                type: 'collapsed',
                data: { count: 1 },
                updatable: false,
                selected: edge.selected,
              });
            }
          }
        });
        if (collapsedEdgesToCreate.length) {
          state.edges = applyEdgeChanges(
            collapsedEdgesToCreate.map((edge) => ({ type: 'add', item: edge })),
            state.edges
          );
        }
      }
    },
    nodeLabelChanged: (state, action: PayloadAction<{ nodeId: string; label: string }>) => {
      const { nodeId, label } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (isInvocationNode(node) || isNotesNode(node)) {
        node.data.label = label;
      }
    },
    nodeNotesChanged: (state, action: PayloadAction<{ nodeId: string; notes: string }>) => {
      const { nodeId, notes } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      node.data.notes = notes;
    },
    fieldValueReset: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, zStatefulFieldValue);
    },
    fieldStringValueChanged: (state, action: FieldValueAction<StringFieldValue>) => {
      fieldValueReducer(state, action, zStringFieldValue);
    },
    fieldNumberValueChanged: (state, action: FieldValueAction<IntegerFieldValue | FloatFieldValue>) => {
      fieldValueReducer(state, action, zIntegerFieldValue.or(zFloatFieldValue));
    },
    fieldBooleanValueChanged: (state, action: FieldValueAction<BooleanFieldValue>) => {
      fieldValueReducer(state, action, zBooleanFieldValue);
    },
    fieldBoardValueChanged: (state, action: FieldValueAction<BoardFieldValue>) => {
      fieldValueReducer(state, action, zBoardFieldValue);
    },
    fieldImageValueChanged: (state, action: FieldValueAction<ImageFieldValue>) => {
      fieldValueReducer(state, action, zImageFieldValue);
    },
    fieldColorValueChanged: (state, action: FieldValueAction<ColorFieldValue>) => {
      fieldValueReducer(state, action, zColorFieldValue);
    },
    fieldMainModelValueChanged: (state, action: FieldValueAction<MainModelFieldValue>) => {
      fieldValueReducer(state, action, zMainModelFieldValue);
    },
    fieldModelIdentifierValueChanged: (state, action: FieldValueAction<ModelIdentifierFieldValue>) => {
      fieldValueReducer(state, action, zModelIdentifierFieldValue);
    },
    fieldRefinerModelValueChanged: (state, action: FieldValueAction<SDXLRefinerModelFieldValue>) => {
      fieldValueReducer(state, action, zSDXLRefinerModelFieldValue);
    },
    fieldVaeModelValueChanged: (state, action: FieldValueAction<VAEModelFieldValue>) => {
      fieldValueReducer(state, action, zVAEModelFieldValue);
    },
    fieldLoRAModelValueChanged: (state, action: FieldValueAction<LoRAModelFieldValue>) => {
      fieldValueReducer(state, action, zLoRAModelFieldValue);
    },
    fieldControlNetModelValueChanged: (state, action: FieldValueAction<ControlNetModelFieldValue>) => {
      fieldValueReducer(state, action, zControlNetModelFieldValue);
    },
    fieldIPAdapterModelValueChanged: (state, action: FieldValueAction<IPAdapterModelFieldValue>) => {
      fieldValueReducer(state, action, zIPAdapterModelFieldValue);
    },
    fieldT2IAdapterModelValueChanged: (state, action: FieldValueAction<T2IAdapterModelFieldValue>) => {
      fieldValueReducer(state, action, zT2IAdapterModelFieldValue);
    },
    fieldSpandrelImageToImageModelValueChanged: (
      state,
      action: FieldValueAction<SpandrelImageToImageModelFieldValue>
    ) => {
      fieldValueReducer(state, action, zSpandrelImageToImageModelFieldValue);
    },
    fieldT5EncoderValueChanged: (state, action: FieldValueAction<T5EncoderModelFieldValue>) => {
      fieldValueReducer(state, action, zT5EncoderModelFieldValue);
    },
    fieldCLIPEmbedValueChanged: (state, action: FieldValueAction<CLIPEmbedModelFieldValue>) => {
      fieldValueReducer(state, action, zCLIPEmbedModelFieldValue);
    },
    fieldFluxVAEModelValueChanged: (state, action: FieldValueAction<FluxVAEModelFieldValue>) => {
      fieldValueReducer(state, action, zFluxVAEModelFieldValue);
    },
    fieldEnumModelValueChanged: (state, action: FieldValueAction<EnumFieldValue>) => {
      fieldValueReducer(state, action, zEnumFieldValue);
    },
    fieldSchedulerValueChanged: (state, action: FieldValueAction<SchedulerFieldValue>) => {
      fieldValueReducer(state, action, zSchedulerFieldValue);
    },
    notesNodeValueChanged: (state, action: PayloadAction<{ nodeId: string; value: string }>) => {
      const { nodeId, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isNotesNode(node)) {
        return;
      }
      node.data.notes = value;
    },
    nodeEditorReset: (state) => {
      state.nodes = [];
      state.edges = [];
    },
    undo: (state) => state,
    redo: (state) => state,
  },
  extraReducers: (builder) => {
    builder.addCase(workflowLoaded, (state, action) => {
      const { nodes, edges } = action.payload;
      state.nodes = applyNodeChanges(
        nodes.map((node) => ({
          type: 'add',
          item: { ...node, ...SHARED_NODE_PROPERTIES },
        })),
        []
      );
      state.edges = applyEdgeChanges(
        edges.map((edge) => ({ type: 'add', item: edge })),
        []
      );
    });
  },
});

export const {
  edgesChanged,
  fieldValueReset,
  fieldBoardValueChanged,
  fieldBooleanValueChanged,
  fieldColorValueChanged,
  fieldControlNetModelValueChanged,
  fieldEnumModelValueChanged,
  fieldImageValueChanged,
  fieldIPAdapterModelValueChanged,
  fieldT2IAdapterModelValueChanged,
  fieldSpandrelImageToImageModelValueChanged,
  fieldLabelChanged,
  fieldLoRAModelValueChanged,
  fieldModelIdentifierValueChanged,
  fieldMainModelValueChanged,
  fieldNumberValueChanged,
  fieldRefinerModelValueChanged,
  fieldSchedulerValueChanged,
  fieldStringValueChanged,
  fieldVaeModelValueChanged,
  fieldT5EncoderValueChanged,
  fieldCLIPEmbedValueChanged,
  fieldFluxVAEModelValueChanged,
  nodeEditorReset,
  nodeIsIntermediateChanged,
  nodeIsOpenChanged,
  nodeLabelChanged,
  nodeNotesChanged,
  nodesChanged,
  nodeUseCacheChanged,
  notesNodeValueChanged,
  undo,
  redo,
} = nodesSlice.actions;

export const $cursorPos = atom<XYPosition | null>(null);
export const $templates = atom<Templates>({});
export const $hasTemplates = computed($templates, (templates) => Object.keys(templates).length > 0);
export const $copiedNodes = atom<AnyNode[]>([]);
export const $copiedEdges = atom<InvocationNodeEdge[]>([]);
export const $edgesToCopiedNodes = atom<InvocationNodeEdge[]>([]);
export const $pendingConnection = atom<PendingConnection | null>(null);
export const $edgePendingUpdate = atom<Edge | null>(null);
export const $didUpdateEdge = atom(false);
export const $lastEdgeUpdateMouseEvent = atom<MouseEvent | null>(null);

export const $viewport = atom<Viewport>({ x: 0, y: 0, zoom: 1 });
export const [useAddNodeCmdk, $addNodeCmdk] = buildUseBoolean(false);

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateNodesState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const nodesPersistConfig: PersistConfig<NodesState> = {
  name: nodesSlice.name,
  initialState: initialNodesState,
  migrate: migrateNodesState,
  persistDenylist: [],
};

const isSelectionAction = (action: UnknownAction) => {
  if (nodesChanged.match(action)) {
    if (action.payload.every((change) => change.type === 'select')) {
      return true;
    }
  }
  if (edgesChanged.match(action)) {
    if (action.payload.every((change) => change.type === 'select')) {
      return true;
    }
  }
  return false;
};

const individualGroupByMatcher = isAnyOf(nodesChanged);

export const nodesUndoableConfig: UndoableOptions<NodesState, UnknownAction> = {
  limit: 64,
  undoType: nodesSlice.actions.undo.type,
  redoType: nodesSlice.actions.redo.type,
  groupBy: (action, state, history) => {
    if (isSelectionAction(action)) {
      // Changes to selection should never be recorded on their own
      return history.group;
    }
    if (individualGroupByMatcher(action)) {
      return action.type;
    }
    return null;
  },
  filter: (action, _state, _history) => {
    // Ignore all actions from other slices
    if (!action.type.startsWith(nodesSlice.name)) {
      return false;
    }
    if (nodesChanged.match(action)) {
      if (action.payload.every((change) => change.type === 'dimensions')) {
        return false;
      }
    }
    return true;
  },
};

// This is used for tracking `state.workflow.isTouched`
export const isAnyNodeOrEdgeMutation = isAnyOf(
  edgesChanged,
  fieldBoardValueChanged,
  fieldBooleanValueChanged,
  fieldColorValueChanged,
  fieldControlNetModelValueChanged,
  fieldEnumModelValueChanged,
  fieldImageValueChanged,
  fieldIPAdapterModelValueChanged,
  fieldT2IAdapterModelValueChanged,
  fieldLabelChanged,
  fieldLoRAModelValueChanged,
  fieldMainModelValueChanged,
  fieldNumberValueChanged,
  fieldRefinerModelValueChanged,
  fieldSchedulerValueChanged,
  fieldStringValueChanged,
  fieldVaeModelValueChanged,
  fieldT5EncoderValueChanged,
  fieldCLIPEmbedValueChanged,
  fieldFluxVAEModelValueChanged,
  // The `nodesChanged` has extra logic and is handled in its own extra reducer
  // nodesChanged,
  nodeIsIntermediateChanged,
  nodeIsOpenChanged,
  nodeLabelChanged,
  nodeNotesChanged,
  nodeUseCacheChanged,
  notesNodeValueChanged
);
