import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { workflowLoaded } from 'features/nodes/store/actions';
import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type {
  BoardFieldValue,
  BooleanFieldValue,
  ColorFieldValue,
  ControlNetModelFieldValue,
  EnumFieldValue,
  FieldValue,
  FloatFieldValue,
  ImageFieldValue,
  IntegerFieldValue,
  IPAdapterModelFieldValue,
  LoRAModelFieldValue,
  MainModelFieldValue,
  SchedulerFieldValue,
  SDXLRefinerModelFieldValue,
  StatefulFieldValue,
  StringFieldValue,
  T2IAdapterModelFieldValue,
  VAEModelFieldValue,
} from 'features/nodes/types/field';
import {
  zBoardFieldValue,
  zBooleanFieldValue,
  zColorFieldValue,
  zControlNetModelFieldValue,
  zEnumFieldValue,
  zFloatFieldValue,
  zImageFieldValue,
  zIntegerFieldValue,
  zIPAdapterModelFieldValue,
  zLoRAModelFieldValue,
  zMainModelFieldValue,
  zSchedulerFieldValue,
  zSDXLRefinerModelFieldValue,
  zStatefulFieldValue,
  zStringFieldValue,
  zT2IAdapterModelFieldValue,
  zVAEModelFieldValue,
} from 'features/nodes/types/field';
import type { AnyNode, InvocationNodeEdge, NodeExecutionState } from 'features/nodes/types/invocation';
import { isInvocationNode, isNotesNode, zNodeStatus } from 'features/nodes/types/invocation';
import { forEach } from 'lodash-es';
import { atom } from 'nanostores';
import type { Connection, Edge, EdgeChange, EdgeRemoveChange, Node, NodeChange, Viewport, XYPosition } from 'reactflow';
import { addEdge, applyEdgeChanges, applyNodeChanges, getConnectedEdges, getIncomers, getOutgoers } from 'reactflow';
import type { UndoableOptions } from 'redux-undo';
import {
  socketGeneratorProgress,
  socketInvocationComplete,
  socketInvocationError,
  socketInvocationStarted,
  socketQueueItemStatusChanged,
} from 'services/events/actions';
import type { z } from 'zod';

import type { NodesState, PendingConnection, Templates } from './types';
import { findUnoccupiedPosition } from './util/findUnoccupiedPosition';

const initialNodeExecutionState: Omit<NodeExecutionState, 'nodeId'> = {
  status: zNodeStatus.enum.PENDING,
  error: null,
  progress: null,
  progressImage: null,
  outputs: [],
};

const initialNodesState: NodesState = {
  _version: 1,
  nodes: [],
  edges: [],
  nodeExecutionStates: {},
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
    },
    nodeReplaced: (state, action: PayloadAction<{ nodeId: string; node: Node }>) => {
      const nodeIndex = state.nodes.findIndex((n) => n.id === action.payload.nodeId);
      if (nodeIndex < 0) {
        return;
      }
      state.nodes[nodeIndex] = action.payload.node;
    },
    nodeAdded: (state, action: PayloadAction<{ node: AnyNode; cursorPos: XYPosition | null }>) => {
      const { node, cursorPos } = action.payload;
      const position = findUnoccupiedPosition(
        state.nodes,
        cursorPos?.x ?? node.position.x,
        cursorPos?.y ?? node.position.y
      );
      node.position = position;
      node.selected = true;

      state.nodes = applyNodeChanges(
        state.nodes.map((n) => ({ id: n.id, type: 'select', selected: false })),
        state.nodes
      );

      state.edges = applyEdgeChanges(
        state.edges.map((e) => ({ id: e.id, type: 'select', selected: false })),
        state.edges
      );

      state.nodes.push(node);

      if (!isInvocationNode(node)) {
        return;
      }

      state.nodeExecutionStates[node.id] = {
        nodeId: node.id,
        ...initialNodeExecutionState,
      };
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange[]>) => {
      state.edges = applyEdgeChanges(action.payload, state.edges);
    },
    edgeAdded: (state, action: PayloadAction<Edge>) => {
      state.edges = addEdge(action.payload, state.edges);
    },
    connectionMade: (state, action: PayloadAction<Connection>) => {
      state.edges = addEdge({ ...action.payload, type: 'default' }, state.edges);
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
    edgeDeleted: (state, action: PayloadAction<string>) => {
      state.edges = state.edges.filter((e) => e.id !== action.payload);
    },
    edgesDeleted: (state, action: PayloadAction<Edge[]>) => {
      const edges = action.payload;
      const collapsedEdges = edges.filter((e) => e.type === 'collapsed');

      // if we delete a collapsed edge, we need to delete all collapsed edges between the same nodes
      if (collapsedEdges.length) {
        const edgeChanges: EdgeRemoveChange[] = [];
        collapsedEdges.forEach((collapsedEdge) => {
          state.edges.forEach((edge) => {
            if (edge.source === collapsedEdge.source && edge.target === collapsedEdge.target) {
              edgeChanges.push({ id: edge.id, type: 'remove' });
            }
          });
        });
        state.edges = applyEdgeChanges(edgeChanges, state.edges);
      }
    },
    nodesDeleted: (state, action: PayloadAction<AnyNode[]>) => {
      action.payload.forEach((node) => {
        if (!isInvocationNode(node)) {
          return;
        }
        delete state.nodeExecutionStates[node.id];
      });
    },
    nodeLabelChanged: (state, action: PayloadAction<{ nodeId: string; label: string }>) => {
      const { nodeId, label } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      node.data.label = label;
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
    nodeExclusivelySelected: (state, action: PayloadAction<string>) => {
      const nodeId = action.payload;
      state.nodes = applyNodeChanges(
        state.nodes.map((n) => ({
          id: n.id,
          type: 'select',
          selected: n.id === nodeId ? true : false,
        })),
        state.nodes
      );
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
    selectedAll: (state) => {
      state.nodes = applyNodeChanges(
        state.nodes.map((n) => ({ id: n.id, type: 'select', selected: true })),
        state.nodes
      );
      state.edges = applyEdgeChanges(
        state.edges.map((e) => ({ id: e.id, type: 'select', selected: true })),
        state.edges
      );
    },
    selectionPasted: (state, action: PayloadAction<{ nodes: AnyNode[]; edges: InvocationNodeEdge[] }>) => {
      const { nodes, edges } = action.payload;

      const nodeChanges: NodeChange[] = [];

      // Deselect existing nodes
      state.nodes.forEach((n) => {
        nodeChanges.push({
          id: n.data.id,
          type: 'select',
          selected: false,
        });
      });
      // Add new nodes
      nodes.forEach((n) => {
        nodeChanges.push({
          item: n,
          type: 'add',
        });
      });

      const edgeChanges: EdgeChange[] = [];
      // Deselect existing edges
      state.edges.forEach((e) => {
        edgeChanges.push({
          id: e.id,
          type: 'select',
          selected: false,
        });
      });
      // Add new edges
      edges.forEach((e) => {
        edgeChanges.push({
          item: e,
          type: 'add',
        });
      });

      state.nodes = applyNodeChanges(nodeChanges, state.nodes);
      state.edges = applyEdgeChanges(edgeChanges, state.edges);

      // Add node execution states for new nodes
      nodes.forEach((node) => {
        state.nodeExecutionStates[node.id] = {
          nodeId: node.id,
          ...deepClone(initialNodeExecutionState),
        };
      });
    },
    undo: (state) => state,
    redo: (state) => state,
  },
  extraReducers: (builder) => {
    builder.addCase(workflowLoaded, (state, action) => {
      const { nodes, edges } = action.payload;
      state.nodes = applyNodeChanges(
        nodes.map((node) => ({
          item: { ...node, ...SHARED_NODE_PROPERTIES },
          type: 'add',
        })),
        []
      );
      state.edges = applyEdgeChanges(
        edges.map((edge) => ({ item: edge, type: 'add' })),
        []
      );

      state.nodeExecutionStates = nodes.reduce<Record<string, NodeExecutionState>>((acc, node) => {
        acc[node.id] = {
          nodeId: node.id,
          ...initialNodeExecutionState,
        };
        return acc;
      }, {});
    });

    builder.addCase(socketInvocationStarted, (state, action) => {
      const { source_node_id } = action.payload.data;
      const node = state.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = zNodeStatus.enum.IN_PROGRESS;
      }
    });
    builder.addCase(socketInvocationComplete, (state, action) => {
      const { source_node_id, result } = action.payload.data;
      const nes = state.nodeExecutionStates[source_node_id];
      if (nes) {
        nes.status = zNodeStatus.enum.COMPLETED;
        if (nes.progress !== null) {
          nes.progress = 1;
        }
        nes.outputs.push(result);
      }
    });
    builder.addCase(socketInvocationError, (state, action) => {
      const { source_node_id } = action.payload.data;
      const node = state.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = zNodeStatus.enum.FAILED;
        node.error = action.payload.data.error;
        node.progress = null;
        node.progressImage = null;
      }
    });
    builder.addCase(socketGeneratorProgress, (state, action) => {
      const { source_node_id, step, total_steps, progress_image } = action.payload.data;
      const node = state.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = zNodeStatus.enum.IN_PROGRESS;
        node.progress = (step + 1) / total_steps;
        node.progressImage = progress_image ?? null;
      }
    });
    builder.addCase(socketQueueItemStatusChanged, (state, action) => {
      if (['in_progress'].includes(action.payload.data.queue_item.status)) {
        forEach(state.nodeExecutionStates, (nes) => {
          nes.status = zNodeStatus.enum.PENDING;
          nes.error = null;
          nes.progress = null;
          nes.progressImage = null;
          nes.outputs = [];
        });
      }
    });
  },
});

export const {
  connectionMade,
  edgeDeleted,
  edgesChanged,
  edgesDeleted,
  fieldValueReset,
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
  nodeAdded,
  nodeReplaced,
  nodeEditorReset,
  nodeExclusivelySelected,
  nodeIsIntermediateChanged,
  nodeIsOpenChanged,
  nodeLabelChanged,
  nodeNotesChanged,
  nodesChanged,
  nodesDeleted,
  nodeUseCacheChanged,
  notesNodeValueChanged,
  selectedAll,
  selectionPasted,
  edgeAdded,
  undo,
  redo,
} = nodesSlice.actions;

export const $cursorPos = atom<XYPosition | null>(null);
export const $templates = atom<Templates>({});
export const $copiedNodes = atom<AnyNode[]>([]);
export const $copiedEdges = atom<InvocationNodeEdge[]>([]);
export const $pendingConnection = atom<PendingConnection | null>(null);
export const $isModifyingEdge = atom(false);
export const $viewport = atom<Viewport>({ x: 0, y: 0, zoom: 1 });
export const $isAddNodePopoverOpen = atom(false);
export const closeAddNodePopover = () => {
  $isAddNodePopoverOpen.set(false);
  $pendingConnection.set(null);
};
export const openAddNodePopover = () => {
  $isAddNodePopoverOpen.set(true);
};

export const selectNodesSlice = (state: RootState) => state.nodes.present;

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

const selectionMatcher = isAnyOf(selectedAll, selectionPasted, nodeExclusivelySelected);

const isSelectionAction = (action: UnknownAction) => {
  if (selectionMatcher(action)) {
    return true;
  }
  if (nodesChanged.match(action)) {
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
  connectionMade,
  edgeDeleted,
  edgesChanged,
  edgesDeleted,
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
  nodeAdded,
  nodeReplaced,
  nodeIsIntermediateChanged,
  nodeIsOpenChanged,
  nodeLabelChanged,
  nodeNotesChanged,
  nodesDeleted,
  nodeUseCacheChanged,
  notesNodeValueChanged,
  selectionPasted,
  edgeAdded
);
